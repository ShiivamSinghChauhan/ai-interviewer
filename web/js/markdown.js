// Just enough Markdown for the reports: headings, lists, tables, bold, italics and code.
function markdown(src) {
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/(^|[\s(])[*_]([^*_\s][^*_]*?)[*_](?=[\s).,;:!?]|$)/g, "$1<i>$2</i>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  const lines = (src || "").split(/\r?\n/);
  const isList = (l) => /^\s*([-*]|\d+\.)\s+/.test(l);
  let html = "", i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^\s*\|/.test(l)) {
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const body = rows.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r));
      html += "<table>" + body.map((r, j) => "<tr>" + cells(r).map((c) => (j === 0 ? `<th>${inline(c)}</th>` : `<td>${inline(c)}</td>`)).join("") + "</tr>").join("") + "</table>";
      continue;
    }
    const h = l.match(/^(#{1,4})\s+(.*)/);
    if (h) { const n = Math.min(h[1].length + 1, 5); html += `<h${n}>${inline(h[2])}</h${n}>`; i++; continue; }
    if (isList(l)) {
      const tag = /^\s*\d+\./.test(l) ? "ol" : "ul";
      html += `<${tag}>`;
      while (i < lines.length && isList(lines[i])) html += "<li>" + inline(lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, "")) + "</li>";
      html += `</${tag}>`;
      continue;
    }
    if (!l.trim() || /^-{3,}$/.test(l.trim())) { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*\|)/.test(lines[i]) && !isList(lines[i])) para.push(lines[i++]);
    html += "<p>" + inline(para.join(" ")) + "</p>";
  }
  return html;
}
