// Tiny shared helpers used across modules (kept flat; extracted at the
// third duplication per project style).

export function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function tailCap(s: string, n: number): string {
  return s.length > n ? s.slice(-n) : s;
}
