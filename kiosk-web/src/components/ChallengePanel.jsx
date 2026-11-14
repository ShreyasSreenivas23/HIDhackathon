import { ACTION_LABEL } from "../lib/faceTracker";

export default function ChallengePanel({ actions, index, secondsLeft, who }) {
  return (
    <div className="challenge">
      <div className="badge verify"><span className="icon">!</span><span>VERIFY</span></div>
      {who && <p className="who">Verifying: <strong>{who}</strong></p>}
      <p className="message">Quick check — please follow the prompts</p>
      <ol className="steps">
        {actions.map((a, i) => (
          <li key={a} className={i < index ? "done" : i === index ? "active" : ""}>
            {i < index ? "✓ " : ""}{ACTION_LABEL[a] || a}
          </li>
        ))}
      </ol>
      <p className="timer">{secondsLeft}s</p>
    </div>
  );
}
