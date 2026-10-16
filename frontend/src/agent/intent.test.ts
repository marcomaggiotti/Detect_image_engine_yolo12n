import { describe, expect, it } from "vitest";
import { parseIntent } from "./intent";

describe("parseIntent with photos", () => {
  it.each([
    ["", { kind: "check" }],
    ["who is this?", { kind: "check" }],
    ["This is Luna", { kind: "save", name: "luna" }],
    ["this is milo, remember him", { kind: "save", name: "milo" }],
    ["it's Mr Whiskers!", { kind: "save", name: "mr-whiskers" }],
    ["save as luna please", { kind: "save", name: "luna" }],
    ["add them to milo", { kind: "save", name: "milo" }],
    ["remember her as Chloé", { kind: "save", name: "chloe" }],
    ["call him tom", { kind: "save", name: "tom" }],
    ["meet my cat luna", { kind: "save", name: "luna" }],
    ["milo", { kind: "save", name: "milo" }], // a known name on its own
    ["pizza", { kind: "check" }], // not a known name
  ] as const)("%j -> %j", (text, intent) => {
    expect(parseIntent(text, true, ["luna", "milo"])).toEqual(intent);
  });
});

describe("parseIntent without photos", () => {
  it.each([
    ["help", { kind: "help" }],
    ["What can you do?", { kind: "help" }],
    ["How do I add a cat?", { kind: "help" }],
    ["show my cats", { kind: "list" }],
    ["Which cats do you know", { kind: "list" }],
    ["cats", { kind: "list" }],
    ["forget Milo", { kind: "forget", name: "milo" }],
    ["please remove cat luna.", { kind: "forget", name: "luna" }],
    ["hello there", { kind: "unknown" }],
    ["", { kind: "unknown" }],
  ] as const)("%j -> %j", (text, intent) => {
    expect(parseIntent(text, false)).toEqual(intent);
  });
});
