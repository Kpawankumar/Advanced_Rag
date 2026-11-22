import { useMemo, useState } from "react";
import { Check, Copy } from "lucide-react";

// Lightweight display-side highlighting: a shared tokenizer with per-family
// keywords. Output is decoration only; it never rewrites the code text.
const KEYWORDS: Record<string, Set<string>> = {
  python: new Set(
    "def return if elif else for while import from as class try except finally with lambda pass break continue yield global nonlocal assert async await raise and or not in is None True False match case".split(
      " ",
    ),
  ),
  javascript: new Set(
    "const let var function return if else for while import export default from as class extends new try catch finally throw async await of in typeof instanceof this null undefined true false yield delete do switch case break continue".split(
      " ",
    ),
  ),
  typescript: new Set(
    "const let var function return if else for while import export default from as class extends implements interface type enum new try catch finally throw async await of in typeof instanceof this null undefined true false readonly public private protected abstract declare keyof infer never unknown any void".split(
      " ",
    ),
  ),
  sql: new Set(
    "SELECT FROM WHERE INSERT INTO VALUES UPDATE SET DELETE CREATE TABLE ALTER DROP JOIN LEFT RIGHT INNER OUTER ON GROUP BY ORDER HAVING LIMIT OFFSET AS AND OR NOT NULL IS IN LIKE BETWEEN DISTINCT COUNT SUM AVG MIN MAX PRIMARY KEY FOREIGN REFERENCES UNION ALL WITH CASE WHEN THEN ELSE END".split(
      " ",
    ),
  ),
  java: new Set(
    "public private protected class interface extends implements static final void int long double float boolean char String new return if else for while try catch finally throw throws import package this super null true false abstract enum record instanceof var".split(
      " ",
    ),
  ),
  c: new Set(
    "int long short char float double void unsigned signed struct union enum typedef static const return if else for while do switch case break continue sizeof malloc free NULL include define".split(
      " ",
    ),
  ),
  bash: new Set(
    "if then else elif fi for while do done case esac function return export local echo read cd ls sudo apt apt-get npm node python pip git set unset trap exit".split(
      " ",
    ),
  ),
  go: new Set(
    "func package import return if else for range switch case break continue defer go chan select type struct interface map const var nil true false string int int64 float64 bool error make new".split(
      " ",
    ),
  ),
  rust: new Set(
    "fn let mut const struct enum impl trait pub use mod match if else for while loop return break continue self Self Some None Ok Err where as dyn ref move async await".split(
      " ",
    ),
  ),
};

const FAMILY = (language: string): string => {
  const name = language.toLowerCase();
  if (["py", "python"].includes(name)) return "python";
  if (["js", "jsx", "javascript", "node", "nodejs", "mjs", "cjs"].includes(name))
    return "javascript";
  if (["ts", "tsx", "typescript"].includes(name)) return "typescript";
  if (["sql", "sqlite", "postgres", "postgresql", "mysql"].includes(name)) return "sql";
  if (["java", "kotlin"].includes(name)) return "java";
  if (["c", "h"].includes(name)) return "c";
  if (["cpp", "c++", "cc", "hpp", "csharp", "cs", "c#"].includes(name)) return "c";
  if (["sh", "bash", "shell", "zsh", "console"].includes(name)) return "bash";
  if (["go", "golang"].includes(name)) return "go";
  if (["rs", "rust"].includes(name)) return "rust";
  return "";
};

const TOKEN = new RegExp(
  [
    "(?<comment>#[^\\n]*|//[^\\n]*|/\\*[\\s\\S]*?\\*/|--[^\\n]*)",
    "(?<string>\"(?:\\\\.|[^\"\\\\])*\"|'(?:\\\\.|[^'\\\\])*'|`(?:\\\\.|[^`\\\\])*`)",
    "(?<number>\\b\\d[\\w.]*\\b)",
    "(?<word>[A-Za-z_][A-Za-z0-9_]*)",
  ].join("|"),
  "g",
);

function highlight(code: string, language: string) {
  const keywords = KEYWORDS[FAMILY(language)];
  const nodes: { text: string; kind: string }[] = [];
  let last = 0;
  for (const match of code.matchAll(TOKEN)) {
    const index = match.index ?? 0;
    if (index > last) nodes.push({ text: code.slice(last, index), kind: "" });
    const groups = match.groups ?? {};
    if (groups.word && keywords?.has(groups.word)) {
      nodes.push({ text: groups.word, kind: "tok-keyword" });
    } else if (groups.comment) {
      nodes.push({ text: groups.comment, kind: "tok-comment" });
    } else if (groups.string) {
      nodes.push({ text: groups.string, kind: "tok-string" });
    } else if (groups.number) {
      nodes.push({ text: groups.number, kind: "tok-number" });
    } else {
      nodes.push({ text: match[0], kind: "" });
    }
    last = index + match[0].length;
  }
  if (last < code.length) nodes.push({ text: code.slice(last), kind: "" });
  return nodes;
}

export function CodeBlock({ code, language }: { code: string; language: string }) {
  const [copied, setCopied] = useState(false);
  const nodes = useMemo(() => highlight(code, language), [code, language]);
  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }
  return (
    <div className="code-block">
      <div className="code-head">
        <span>{language || "code"}</span>
        <button onClick={() => void copy()} aria-label="Copy code">
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code>
          {nodes.map((node, i) =>
            node.kind ? (
              <span className={node.kind} key={i}>
                {node.text}
              </span>
            ) : (
              node.text
            ),
          )}
        </code>
      </pre>
    </div>
  );
}

// The fenced block that starts a code section, including its language tag.
const INLINE_CODE = /`([^`\n]+)`/g;
const BOLD = /\*\*([^*\n]+)\*\*/g;
const LEADING_HEADING = /^#{1,6} /;
const BULLET = /^\s*(?:[-*•]|\d+[.)]) /;

type InlinePart = { kind: "text" | "code" | "bold"; value: string; key: string };

function inlineSegments(text: string, keyBase: string): InlinePart[] {
  // Inline `code` and **bold** spans; anything else renders as plain text.
  const parts: InlinePart[] = [];
  const marks: { start: number; end: number; kind: "code" | "bold"; inner: string }[] = [];
  for (const match of text.matchAll(INLINE_CODE)) {
    const start = match.index ?? 0;
    marks.push({ start, end: start + match[0].length, kind: "code", inner: match[1] });
  }
  for (const match of text.matchAll(BOLD)) {
    const start = match.index ?? 0;
    const overlaps = marks.some(
      (mark) => start < mark.end && mark.start < start + match[0].length,
    );
    if (!overlaps) {
      marks.push({ start, end: start + match[0].length, kind: "bold", inner: match[1] });
    }
  }
  marks.sort((a, b) => a.start - b.start);
  let cursor = 0;
  for (const mark of marks) {
    if (mark.start > cursor) {
      parts.push({ kind: "text", value: text.slice(cursor, mark.start), key: `${keyBase}:t${cursor}` });
    }
    parts.push({ kind: mark.kind, value: mark.inner, key: `${keyBase}:${mark.kind}${mark.start}` });
    cursor = mark.end;
  }
  if (cursor < text.length) {
    parts.push({ kind: "text", value: text.slice(cursor), key: `${keyBase}:t${cursor}` });
  }
  return parts;
}

function inlineText(text: string, keyBase: string) {
  return inlineSegments(text, keyBase).map((part) =>
    part.kind === "text" ? (
      part.value
    ) : part.kind === "code" ? (
      <code className="inline-code" key={part.key}>
        {part.value}
      </code>
    ) : (
      <strong key={part.key}>{part.value}</strong>
    ),
  );
}

/** Renders an assistant answer: fenced code blocks plus lightweight text. */
type Segment = { type: "code"; language: string; code: string } | { type: "text"; text: string };

export function AnswerBody({ answer }: { answer: string }) {
  const segments = useMemo<Segment[]>(() => {
    const result: Segment[] = [];
    let rest = answer;
    while (rest.length) {
      const start = rest.indexOf("```");
      if (start === -1) {
        result.push({ type: "text", text: rest });
        break;
      }
      if (start > 0) result.push({ type: "text", text: rest.slice(0, start) });
      const after = rest.slice(start + 3);
      const header = /^([\w+#.-]*)[ \t]*\r?\n/.exec(after);
      const language = header ? header[1] : "";
      const codeStart = header ? header[0].length : 0;
      const end = after.indexOf("```", codeStart);
      if (end === -1) {
        result.push({ type: "code", language, code: after.slice(codeStart) });
        break;
      }
      result.push({ type: "code", language, code: after.slice(codeStart, end) });
      rest = after.slice(end + 3);
      // A closing fence is often followed by its own line break.
      rest = rest.replace(/^\r?\n/, "");
    }
    return result;
  }, [answer]);

  return (
    <div className="answer-body">
      {segments.map((segment, i) => {
        if (segment.type === "code") {
          return <CodeBlock code={segment.code} language={segment.language} key={i} />;
        }
        const lines = segment.text.split("\n");
        return (
          <div key={i}>
            {lines.map((line, j) => {
              const trimmed = line.trim();
              if (!trimmed) return null;
              if (BULLET.test(trimmed)) {
                return (
                  <p className="answer-bullet" key={`${i}:${j}`}>
                    {inlineText(trimmed.replace(BULLET, ""), `${i}:${j}`)}
                  </p>
                );
              }
              const heading = LEADING_HEADING.test(trimmed)
                ? trimmed.replace(LEADING_HEADING, "")
                : "";
              return (
                <p className={heading ? "answer-heading" : undefined} key={`${i}:${j}`}>
                  {inlineText(heading || trimmed, `${i}:${j}`)}
                </p>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
