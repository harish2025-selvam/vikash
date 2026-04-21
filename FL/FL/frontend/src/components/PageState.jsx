export function LoadingState({ label = "Loading..." }) {
  return (
    <div className="state-card">
      <div className="state-spinner" />
      <p>{label}</p>
    </div>
  );
}

export function ErrorState({ message }) {
  return (
    <div className="state-card state-card-error">
      <p>{message}</p>
    </div>
  );
}
