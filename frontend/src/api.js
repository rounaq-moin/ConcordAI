const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const TOKEN_KEY = "concordai_auth_token";

function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setStoredToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(token = getStoredToken()) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function readJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      data?.error?.message ||
      data?.message ||
      data?.detail ||
      `Request failed: ${response.status}`;
    const error = new Error(message);
    error.payload = data;
    error.code = data?.error?.code || data?.code || data?.error;
    error.status = response.status;
    throw error;
  }
  return data;
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return readJson(response);
}

export async function signUp(payload) {
  const response = await fetch(`${API_BASE}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await readJson(response);
  setStoredToken(data.token);
  return data;
}

export async function signIn(payload) {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await readJson(response);
  setStoredToken(data.token);
  return data;
}

export async function getMe() {
  const response = await fetch(`${API_BASE}/auth/me`, {
    headers: authHeaders(),
  });
  return readJson(response);
}

export async function signOut() {
  const token = getStoredToken();
  if (token) {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      headers: authHeaders(token),
    }).catch(() => {});
  }
  setStoredToken(null);
}

export async function getHistory() {
  const response = await fetch(`${API_BASE}/history`, {
    headers: authHeaders(),
  });
  return readJson(response);
}

export async function mediate(payload) {
  const response = await fetch(`${API_BASE}/mediate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
  });
  return readJson(response);
}

export async function resolveConversation(conversationId, payload) {
  const response = await fetch(`${API_BASE}/conversations/${conversationId}/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(payload),
  });
  return readJson(response);
}

export { API_BASE, getStoredToken, setStoredToken };
