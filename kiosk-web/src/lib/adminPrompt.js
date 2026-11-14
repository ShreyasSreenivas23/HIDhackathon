// Masked admin-password dialog (window.prompt shows what you type, so it is not used for passwords).
// Resolves with the entered password, or "" if cancelled.
export function askAdminPassword(title = "Admin password") {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "pw-overlay";
    overlay.innerHTML = `
      <form class="pw-box" autocomplete="off">
        <h2></h2>
        <input type="password" name="pw" placeholder="Admin password" autocomplete="current-password" />
        <div class="pw-actions">
          <button type="button" class="link pw-cancel">Cancel</button>
          <button type="submit" class="pw-ok">Unlock</button>
        </div>
      </form>`;
    overlay.querySelector("h2").textContent = title;
    const input = overlay.querySelector("input");
    const done = (value) => { overlay.remove(); resolve(value); };
    overlay.querySelector("form").addEventListener("submit", (e) => { e.preventDefault(); done(input.value); });
    overlay.querySelector(".pw-cancel").addEventListener("click", () => done(""));
    overlay.addEventListener("keydown", (e) => { if (e.key === "Escape") done(""); });
    document.body.appendChild(overlay);
    setTimeout(() => input.focus(), 0);
  });
}
