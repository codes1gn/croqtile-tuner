const STATUS_STYLES: Record<string, string> = {
  pending: "bg-gray-600 text-gray-100",
  waiting: "bg-slate-600 text-slate-200 border border-slate-500",
  running: "bg-blue-600 text-white animate-pulse",
  completed: "bg-emerald-600 text-white",
  cancelled: "bg-yellow-600 text-white",
};

export function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? "bg-gray-500 text-white";
  return (
    <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wide ${style}`}>
      {status}
    </span>
  );
}
