# AWS deployment (cloud mode), P3

The live demo runs in **edge mode** on the laptop. This page shows that the same container runs in the cloud.
Budget: about $0.18/h for `c7i.xlarge` (CPU is enough for these models), plus DynamoDB/S3 pennies. That's well under $40 for the event.
Replace the `<...>` placeholders. Region in the examples: `us-east-1`.

## 1. Storage
```bash
aws dynamodb create-table --table-name hid-enrollments \
  --attribute-definitions AttributeName=user_id,AttributeType=S \
  --key-schema AttributeName=user_id,KeyType=HASH --billing-mode PAY_PER_REQUEST
aws s3 mb s3://hid-audit-<team>
```

## 2. Image → ECR
```bash
aws ecr create-repository --repository-name hid-api
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com
docker build -t hid-api api/
docker tag hid-api <acct>.dkr.ecr.us-east-1.amazonaws.com/hid-api:latest
docker push <acct>.dkr.ecr.us-east-1.amazonaws.com/hid-api:latest
```

## 3. EC2 (simplest path)
- Launch `c7i.xlarge` with Amazon Linux 2023. Attach an **IAM role** with DynamoDB (table), S3 (bucket) and ECR read access. Open port 8000 only to your IP, or put an ALB in front.
- On the instance:
```bash
sudo dnf install -y docker && sudo systemctl start docker
aws ecr get-login-password | sudo docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-1.amazonaws.com
# copy api/models/ to the instance (scp), then:
sudo docker run -d -p 8000:8000 -v /home/ec2-user/models:/srv/models \
  -e STORE=dynamo -e DYNAMO_TABLE=hid-enrollments -e AUDIT_BUCKET=hid-audit-<team> -e AWS_REGION=us-east-1 \
  <acct>.dkr.ecr.us-east-1.amazonaws.com/hid-api:latest
```
- Point the kiosk at it: `VITE_API_TARGET=http://<ec2-ip>:8000 npm run dev`

## 4. Observability
Container logs carry one `DECISION {...}` JSON line per request. Ship them with the CloudWatch agent or the awslogs Docker log driver, then add **metric filters** for decision counts and `total_ms`. That gives a latency and decision dashboard for the slides.

## Scaling story (slide)
The decision service is **stateless**: the gallery is in DynamoDB and audit data in S3. So it can run as **ECS Fargate behind an ALB with autoscaling** on CPU or request count. Each kiosk can also run the same image locally (edge) for offline operation and privacy, syncing templates from DynamoDB. Challenge and OTP state are in memory today; moving them to DynamoDB with a TTL is enough for multi-instance deployments.

**Optional baseline:** run Amazon Rekognition Face Liveness / CompareFaces on the same live test set, and compare APCER and latency on one slide.
