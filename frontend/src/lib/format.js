export const fmt = (v, digits = 1) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};

export const fmtInt = (v) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString();
};

export const prettyDate = (iso) => {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  } catch {
    return iso;
  }
};

// Mean of a numeric metric across summary rows.
export const meanBy = (rows, key) => {
  const vals = rows.map((r) => r[key]).filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
};

export const sumBy = (rows, key) =>
  rows.reduce((a, r) => a + (Number(r[key]) || 0), 0);
