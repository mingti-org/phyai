export function before(rows, t) {
  let lo = 0,
    hi = rows.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (rows[m].t <= t) lo = m + 1;
    else hi = m;
  }
  return rows[lo - 1] ?? null;
}
export function joints(row, side) {
  const v = row?.values ?? {};
  if (v[`joint_position_${side}`]) return v[`joint_position_${side}`];
  if (v.arm_positions)
    return v.arm_positions.slice(
      side === "left" ? 0 : 7,
      side === "left" ? 7 : 14,
    );
  if (v.state)
    return v.state.slice(side === "left" ? 0 : 8, side === "left" ? 7 : 15);
  return null;
}
export function alignedError(feedback, commands, side, index) {
  return feedback.flatMap((row) => {
    const command = before(commands, row.t),
      q = joints(row, side),
      a = joints(command, side);
    return command && row.t - command.t < 0.15 && q && a
      ? [[row.t, a[index] - q[index]]]
      : [];
  });
}
export function bounds(points, fallback = [-1, 1]) {
  const values = points
    .flat()
    .filter((p) => Array.isArray(p) && Number.isFinite(p[1]))
    .map((p) => p[1]);
  if (!values.length) return fallback;
  const low = values.reduce((a, b) => Math.min(a, b), Infinity),
    high = values.reduce((a, b) => Math.max(a, b), -Infinity),
    pad = Math.max(0.01, (high - low) * 0.12);
  return [low - pad, high + pad];
}
