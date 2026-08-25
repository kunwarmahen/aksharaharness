/* md.js — a hand-rolled markdown → HTML renderer for assistant replies.
   No dependency and no build step, for the same reasons the rest of this
   page is vanilla: the server may not fetch anything external, and a
   150-line parser teaches more than a 40 kB library.

   Security model, stated once: NOTHING from the model is ever trusted
   as markup. The text is HTML-escaped FIRST (so <script> becomes text),
   and every tag below is one WE generate around content that is already
   escaped. Link URLs must look like http(s)/mailto or they degrade to
   plain text. */

"use strict";

function escMd(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* ---- inline: code spans first (protected), then emphasis/links ------- */

function inlineMd(text) {
  const codes = [];
  let t = escMd(text);

  // `code spans` — extracted before emphasis can mangle their contents
  t = t.replace(/`([^`\n]+)`/g, (_, body) => {
    codes.push(`<code>${body.trim()}</code>`);
    return `\x00${codes.length - 1}\x00`;
  });

  // images aren't rendered (a chat transcript shouldn't fetch URLs);
  // ![alt](src) degrades to its alt text
  t = t.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, "_$1_");

  // links — scheme-checked; anything else stays visible text
  t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (all, label, url) => {
    const clean = url.replace(/&amp;/g, "&");
    if (!/^(https?:\/\/|mailto:)/i.test(clean)) return all;
    return `<a href="${escMd(clean)}" target="_blank" rel="noreferrer">${label}</a>`;
  });

  t = t.replace(/\*\*\*([^*\n]+)\*\*\*/g, "<strong><em>$1</em></strong>");
  t = t.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  // single asterisks/underscores need a non-word char before the opener
  // so multiplication ("2*3*4") and snake_case stay untouched
  t = t.replace(/(^|[^\w*])\*([^*\n]+)\*(?=$|[^\w*])/g, "$1<em>$2</em>");
  t = t.replace(/(^|[^\w])_([^_\n]+)_(?=$|[^\w])/g, "$1<em>$2</em>");
  t = t.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");

  return t.replace(/\x00(\d+)\x00/g, (_, i) => codes[Number(i)]);
}

/* ---- blocks ----------------------------------------------------------- */

const FENCE_RE = /^\s{0,3}```\s*(\S*)\s*$/;
const HEADING_RE = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const HR_RE = /^\s{0,3}([-*_])\s*(?:\1\s*){2,}$/;
const BULLET_RE = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const QUOTE_RE = /^\s{0,3}> ?(.*)$/;

function splitTableRow(line) {
  const bare = line.trim().replace(/^\|/, "").replace(/\|\s*$/, "");
  return bare.split("|").map((c) => c.trim());
}

function isTableSeparator(cells) {
  return cells.length > 0 &&
    cells.every((c) => /^:?-{3,}:?$/.test(c));
}

/* Lists recurse by indentation. Each item owns every following line
   indented at least to its content column; uniform slicing there keeps
   RELATIVE indentation intact, so deeper bullets recurse naturally. */
function parseList(lines, i) {
  const first = BULLET_RE.exec(lines[i]);
  const base = first[1].length;
  const col = first[1].length + first[2].length + 1; // where content starts
  const ordered = /\d/.test(first[2][0]);
  const items = [];

  while (i < lines.length && lines[i].trim()) {
    const m = BULLET_RE.exec(lines[i]);
    if (!m || m[1].length < base) break;
    const content = [m[3]];
    i += 1;
    while (i < lines.length) {
      const l = lines[i];
      if (!l.trim()) {
        // a blank continues the item only if deeper content follows it
        let k = i + 1;
        while (k < lines.length && !lines[k].trim()) k += 1;
        if (k < lines.length && countIndent(lines[k]) >= col) {
          content.push(""); i += 1; continue;
        }
        break;
      }
      const ind = countIndent(l);
      if (ind < col && BULLET_RE.test(l) && ind <= base) break; // sibling
      if (ind < col && !BULLET_RE.test(l)) break;               // new block
      content.push(l.slice(Math.min(col, l.length)));
      i += 1;
    }

    // prose first; once a sliced-off nested bullet appears, recurse
    let j = 0;
    while (j < content.length && !BULLET_RE.test(content[j])) j += 1;
    const prose = inlineMd(content.slice(0, j).join("\n").trim())
      .replace(/\n/g, "<br>");
    const nested = j < content.length
      ? parseList(content.slice(j), 0).html : "";
    items.push(`<li>${prose}${nested}</li>`);
  }

  const tag = ordered ? "ol" : "ul";
  return { html: `<${tag}>${items.join("")}</${tag}>`, next: i };
}

function countIndent(line) {
  return line.length - line.trimStart().length;
}

function renderMarkdown(src) {
  if (!src || !String(src).trim()) return "";
  const lines = String(src).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) { i += 1; continue; }

    const fence = FENCE_RE.exec(line);
    if (fence) {
      const lang = fence[1] ? ` class="lang-${escMd(fence[1])}"` : "";
      const body = [];
      i += 1;
      while (i < lines.length && !FENCE_RE.test(lines[i])) {
        body.push(lines[i]); i += 1;
      }
      i += 1; // past the closing fence (or EOF — render what we have)
      out.push(`<pre class="md-code"${lang}><code>${escMd(body.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = HEADING_RE.exec(line);
    if (heading) {
      // two levels down: a chat message is not a document outline, so
      // '#' renders like a section title rather than a page title
      const level = Math.min(heading[1].length + 2, 6);
      out.push(`<h${level}>${inlineMd(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (HR_RE.test(line)) { out.push("<hr>"); i += 1; continue; }

    const quote = QUOTE_RE.exec(line);
    if (quote) {
      const body = [quote[1]];
      i += 1;
      while (i < lines.length) {
        const qm = QUOTE_RE.exec(lines[i]);
        if (qm) { body.push(qm[1]); i += 1; continue; }
        if (lines[i].trim() && !BLOCK_START_RE.test(lines[i])) {
          body.push(lines[i].trim()); i += 1; continue; // lazy continuation
        }
        break;
      }
      out.push(`<blockquote>${renderMarkdown(body.join("\n"))}</blockquote>`);
      continue;
    }

    // pipe table: header row, then a ---|--- separator, then rows
    if (line.includes("|") && i + 1 < lines.length) {
      const head = splitTableRow(line);
      const sep = splitTableRow(lines[i + 1]);
      if (head.length > 1 && isTableSeparator(sep)) {
        const cols = head.length;
        const rows = [];
        i += 2;
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          const cells = splitTableRow(lines[i]);
          while (cells.length < cols) cells.push("");
          rows.push(cells.slice(0, cols));
          i += 1;
        }
        const th = head.map((c) => `<th>${inlineMd(c)}</th>`).join("");
        const tb = rows.map((r) =>
          `<tr>${r.map((c) => `<td>${inlineMd(c)}</td>`).join("")}</tr>`).join("");
        out.push(`<div class="table-wrap"><table><thead><tr>${th}</tr></thead>` +
                 `<tbody>${tb}</tbody></table></div>`);
        continue;
      }
    }

    if (BULLET_RE.test(line)) {
      const res = parseList(lines, i);
      out.push(res.html);
      i = res.next;
      continue;
    }

    // paragraph: run of ordinary lines; single newlines become <br>
    // because chat answers lean on line breaks more than on reflow
    const para = [line];
    i += 1;
    while (i < lines.length && lines[i].trim()
           && !BLOCK_START_RE.test(lines[i])) {
      para.push(lines[i]); i += 1;
    }
    out.push(`<p>${inlineMd(para.join("\n")).replace(/\n/g, "<br>")}</p>`);
  }

  return out.join("");
}

/* a line that opens one of the block constructs (paragraph collector
   stops here; blockquote's lazy-continuation check uses it too) */
const BLOCK_START_RE = new RegExp([
  FENCE_RE.source, HEADING_RE.source, HR_RE.source,
  BULLET_RE.source, QUOTE_RE.source,
].join("|"));

if (typeof module !== "undefined") module.exports = { renderMarkdown };
