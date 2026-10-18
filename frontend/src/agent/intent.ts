import { CAT_NAME, toCatName } from "../api";

/** What a message asks for. The assistant understands a few plain requests;
 * it is pattern matching, not a language model. */
export type Intent =
  | { kind: "check" }
  | { kind: "save"; name: string }
  | { kind: "list" }
  | { kind: "forget"; name: string }
  | { kind: "help" }
  | { kind: "unknown" };

// "this is Luna", "it's mr whiskers", "save as luna", "add them to milo", "remember him as milo", "call her luna"
const SAVE_AS =
  /\b(?:this is|that is|it is|it's|its|that's|these are|they are|meet|save (?:it|this|them|these)? ?(?:as|to)|add (?:it|this|them|these)? ?(?:as|to)|remember (?:it|this|him|her|them)? ?as|call (?:it|him|her|them)|named?)\s+(.+)/;
// "this is milo, remember him" -> "milo"
const TRAILING = /\s*(?:[,.!?;].*|\s+(?:please|thanks|thank you|remember.*|now))$/;
const FORGET = /\b(?:forget|remove|delete)\s+(?:cat\s+)?(.+)/;
const LIST = /\b(?:my cats|list|show|who do you know|which cats|cats you know)\b|^cats$/;
const HELP = /^(?:help|\?|what can you do|how does (?:it|this) work|how do i add a cat)/;

function nameIn(text: string): string | null {
  const name = toCatName(text.replace(TRAILING, "").replace(/^(?:a cat called|my cat|the cat)\s+/, "").trim());
  return CAT_NAME.test(name) ? name : null;
}

export function parseIntent(raw: string, hasPhotos: boolean, knownNames: string[] = []): Intent {
  const text = raw.trim().toLowerCase();
  if (hasPhotos) {
    if (!text) return { kind: "check" };
    const saved = SAVE_AS.exec(text);
    const name = saved ? nameIn(saved[1]) : null;
    if (name) return { kind: "save", name };
    const bare = nameIn(text);
    if (bare && knownNames.includes(bare)) return { kind: "save", name: bare };
    return { kind: "check" };
  }
  if (!text) return { kind: "unknown" };
  if (HELP.test(text)) return { kind: "help" };
  const forget = FORGET.exec(text);
  const forgetName = forget ? nameIn(forget[1]) : null;
  if (forgetName) return { kind: "forget", name: forgetName };
  if (LIST.test(text)) return { kind: "list" };
  return { kind: "unknown" };
}
