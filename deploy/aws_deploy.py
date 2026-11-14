"""Deploy the kiosk + API to the HID AWS sandbox: EC2 (builds & runs the container) + CloudFront (HTTPS).

App Runner is denied in the sandbox (apprunner:CreateService implicitDeny), so:
  bundle (code, built kiosk, trained DINOv2 head, seed DB) -> S3
  EC2 (Amazon Linux 2023, c7i.xlarge) user-data: docker build + run on :80, data volume /opt/hid-data
  CloudFront -> https://<id>.cloudfront.net  (camera access on phones needs HTTPS)
Idempotent: state in data/aws_deploy_state.json; rerun reuses resources, `--update` re-uploads + redeploys code.

  api/.venv/Scripts/python deploy/aws_deploy.py            # first deploy
  api/.venv/Scripts/python deploy/aws_deploy.py --update   # push new code to the running instance (via SSM/user-data reboot)
  api/.venv/Scripts/python deploy/aws_deploy.py --status
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import boto3

REGION = "us-east-1"
REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "data" / "aws_deploy_state.json"
NAME = "hid-kiosk"
INSTANCE_TYPES = ["c7i.xlarge", "m7i.xlarge", "t3.xlarge"]
CACHING_DISABLED = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
ALL_VIEWER_EXCEPT_HOST = "b689b0a8-53d0-40ab-baf2-68738e2966ac"


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def make_bundle() -> bytes:
    dist = REPO / "kiosk-web" / "dist"
    assert (dist / "index.html").exists(), "build the kiosk first: cd kiosk-web && VITE_API_BASE= npm run build"
    items = {
        "Dockerfile": REPO / "deploy" / "Dockerfile.ec2",
        "api/app": REPO / "api" / "app",
        "api/scripts": REPO / "api" / "scripts",
        "api/requirements-core.txt": REPO / "api" / "requirements-core.txt",
        "api/thresholds.json": REPO / "api" / "thresholds.json",
        "api/models/dino_head.npz": REPO / "api" / "models" / "dino_head.npz",
        "api/enrollments.db": REPO / "api" / "enrollments.db",
        "kiosk-web/dist": dist,
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc, path in items.items():
            if path.exists():
                tar.add(str(path), arcname=arc, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def user_data(bucket: str) -> str:
    return f"""#!/bin/bash
set -x
exec > /var/log/hid-deploy.log 2>&1
dnf install -y docker
systemctl enable --now docker
mkdir -p /opt/hid /opt/hid-data
cd /opt/hid && rm -rf ./* && aws s3 cp s3://{bucket}/bundle.tgz . --region {REGION} && tar xzf bundle.tgz
docker build -t {NAME} . && (docker rm -f hid || true) && \\
docker run -d --restart always -p 80:8080 -v /opt/hid-data:/srv/data --name hid {NAME}
echo DEPLOY_DONE
"""


def ensure_bucket(s3, acct: str) -> str:
    bucket = f"{NAME}-deploy-{acct}"
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
        s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    return bucket


def ensure_instance_profile(iam, bucket: str) -> str:
    role = f"{NAME}-ec2"
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        iam.get_role(RoleName=role)
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(RoleName=role, AssumeRolePolicyDocument=json.dumps(trust), Description="HID kiosk EC2: read deploy bundle")
    iam.put_role_policy(RoleName=role, PolicyName="read-bundle", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": f"arn:aws:s3:::{bucket}/*"}]}))
    try:
        iam.attach_role_policy(RoleName=role, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
    except Exception as e:
        print("  (SSM policy not attached:", e, ")")
    try:
        iam.get_instance_profile(InstanceProfileName=role)
    except iam.exceptions.NoSuchEntityException:
        iam.create_instance_profile(InstanceProfileName=role)
        iam.add_role_to_instance_profile(InstanceProfileName=role, RoleName=role)
        time.sleep(12)   # IAM propagation before RunInstances can use it
    return role


def ensure_sg(ec2) -> str:
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    sgs = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [f"{NAME}-sg"]}, {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    if sgs:
        return sgs[0]["GroupId"]
    sg = ec2.create_security_group(GroupName=f"{NAME}-sg", Description="HID kiosk: HTTP from CloudFront only", VpcId=vpc)["GroupId"]
    pls = ec2.describe_managed_prefix_lists(Filters=[{"Name": "prefix-list-name", "Values": ["com.amazonaws.global.cloudfront.origin-facing"]}])["PrefixLists"]
    perm = {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80}
    if pls:
        perm["PrefixListIds"] = [{"PrefixListId": pls[0]["PrefixListId"], "Description": "CloudFront origin-facing"}]
    else:
        perm["IpRanges"] = [{"CidrIp": "0.0.0.0/0"}]
    ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[perm])
    return sg


def launch(ec2, ssm, sg: str, profile: str, bucket: str) -> str:
    ami = ssm.get_parameter(Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64")["Parameter"]["Value"]
    last = None
    for itype in INSTANCE_TYPES:
        try:
            r = ec2.run_instances(
                ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1,
                IamInstanceProfile={"Name": profile},
                NetworkInterfaces=[{"DeviceIndex": 0, "AssociatePublicIpAddress": True, "Groups": [sg]}],
                BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 30, "VolumeType": "gp3"}}],
                UserData=user_data(bucket),
                TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": NAME}]}],
            )
            print(f"  launched {itype}")
            return r["Instances"][0]["InstanceId"]
        except Exception as e:
            print(f"  {itype} failed: {e}")
            last = e
    raise last


def ensure_cloudfront(cf, origin_dns: str) -> tuple[str, str]:
    cfg = {
        "CallerReference": f"{NAME}-{int(time.time())}",
        "Comment": "HID kiosk HTTPS front",
        "Enabled": True,
        "PriceClass": "PriceClass_100",
        "Origins": {"Quantity": 1, "Items": [{
            "Id": "ec2", "DomainName": origin_dns,
            "CustomOriginConfig": {"HTTPPort": 80, "HTTPSPort": 443, "OriginProtocolPolicy": "http-only",
                                   "OriginReadTimeout": 60, "OriginKeepaliveTimeout": 5,
                                   "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]}}}]},
        "DefaultCacheBehavior": {
            "TargetOriginId": "ec2", "ViewerProtocolPolicy": "redirect-to-https", "Compress": True,
            "AllowedMethods": {"Quantity": 7, "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
                               "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
            "CachePolicyId": CACHING_DISABLED, "OriginRequestPolicyId": ALL_VIEWER_EXCEPT_HOST},
    }
    d = cf.create_distribution(DistributionConfig=cfg)["Distribution"]
    return d["Id"], d["DomainName"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="re-upload bundle and reboot instance to rebuild")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    sess = boto3.Session(region_name=REGION)
    acct = sess.client("sts").get_caller_identity()["Account"]
    s3, iam, ec2, ssm, cf = (sess.client(n) for n in ("s3", "iam", "ec2", "ssm", "cloudfront"))
    st = load_state()

    if args.status:
        print(json.dumps(st, indent=2))
        if st.get("instance_id"):
            i = ec2.describe_instances(InstanceIds=[st["instance_id"]])["Reservations"][0]["Instances"][0]
            print("instance:", i["State"]["Name"], i.get("PublicDnsName"))
        if st.get("cf_id"):
            print("cloudfront:", cf.get_distribution(Id=st["cf_id"])["Distribution"]["Status"])
        return

    bucket = ensure_bucket(s3, acct)
    bundle = make_bundle()
    s3.put_object(Bucket=bucket, Key="bundle.tgz", Body=bundle)
    print(f"bundle uploaded: s3://{bucket}/bundle.tgz ({len(bundle) / 1e6:.1f} MB)")
    st["bucket"] = bucket

    if args.update and st.get("instance_id"):
        # user-data only runs on first boot: re-arm it, then reboot to rebuild with the new bundle
        iid = st["instance_id"]
        ec2.stop_instances(InstanceIds=[iid]); ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
        ud = "Content-Type: multipart/mixed; boundary=\"//\"\nMIME-Version: 1.0\n\n--//\nContent-Type: text/cloud-config; charset=\"us-ascii\"\n\n#cloud-config\ncloud_final_modules:\n- [scripts-user, always]\n\n--//\nContent-Type: text/x-shellscript; charset=\"us-ascii\"\n\n" + user_data(bucket) + "\n--//--\n"
        ec2.modify_instance_attribute(InstanceId=iid, UserData={"Value": ud.encode()})
        ec2.start_instances(InstanceIds=[iid]); ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
        i = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
        st["public_dns"] = i["PublicDnsName"]
        save_state(st)
        print("instance restarted; rebuild takes ~5-8 min. NOTE: public DNS may change ->", st["public_dns"])
        if st.get("cf_id"):
            c = cf.get_distribution_config(Id=st["cf_id"])
            conf, etag = c["DistributionConfig"], c["ETag"]
            if conf["Origins"]["Items"][0]["DomainName"] != st["public_dns"]:
                conf["Origins"]["Items"][0]["DomainName"] = st["public_dns"]
                cf.update_distribution(Id=st["cf_id"], IfMatch=etag, DistributionConfig=conf)
                print("CloudFront origin updated to new instance DNS")
        return

    profile = ensure_instance_profile(iam, bucket)
    sg = ensure_sg(ec2)
    if not st.get("instance_id"):
        st["instance_id"] = launch(ec2, ssm, sg, profile, bucket)
        save_state(st)
    iid = st["instance_id"]
    ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    i = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    st["public_dns"] = i["PublicDnsName"]
    print("instance running:", iid, st["public_dns"])
    if not st.get("cf_id"):
        st["cf_id"], st["cf_domain"] = ensure_cloudfront(cf, st["public_dns"])
    save_state(st)
    print(f"\nHTTPS URL (ready when CloudFront is Deployed, ~5-10 min): https://{st['cf_domain']}")
    print(f"direct (HTTP, no camera on phones): http://{st['public_dns']}")


if __name__ == "__main__":
    main()
