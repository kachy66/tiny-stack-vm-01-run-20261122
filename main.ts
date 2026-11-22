// Tiny stack VM, make.fun ISA v1. One run answers every instance in
// /input/<id>/ (program.bin + input.bin) as /output/<id>/stdout.bin
// + exit.txt. 32-bit wrapping ints, 256-value stack, 64-deep call
// stack, 256 words of memory, 1M step limit, 64K output limit.
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
// [opcode, immediate bytes, popped, pushed] for every opcode.
const OPS: [number, number, number, number][] = [
  [0, 0, 0, 0], [1, 1, 0, 1], [2, 4, 0, 1], [3, 0, 1, 0], [4, 0, 1, 2],
  [5, 0, 2, 2], [6, 0, 2, 3], [7, 0, 3, 3], [16, 0, 2, 1], [17, 0, 2, 1],
  [18, 0, 2, 1], [19, 0, 2, 1], [20, 0, 2, 1], [21, 0, 2, 1], [22, 0, 2, 1],
  [23, 0, 1, 1], [24, 0, 2, 1], [25, 0, 2, 1], [26, 0, 2, 1], [27, 0, 1, 1],
  [28, 0, 2, 1], [29, 0, 2, 1], [30, 0, 2, 1], [32, 0, 2, 1], [33, 0, 2, 1],
  [34, 0, 2, 1], [48, 2, 0, 0], [49, 2, 1, 0], [50, 2, 1, 0], [51, 2, 0, 0],
  [52, 0, 0, 0], [64, 0, 1, 1], [65, 0, 2, 0], [80, 0, 1, 0], [81, 0, 1, 0],
  [82, 0, 0, 1], [83, 0, 1, 0],
];
const KNOWN = new Uint8Array(256);
const IMM = new Uint8Array(256);
const POPS = new Uint8Array(256);
const PUSHES = new Uint8Array(256);
for (const [op, imm, pops, pushes] of OPS) {
  KNOWN[op] = 1; IMM[op] = imm; POPS[op] = pops; PUSHES[op] = pushes;
}
const enc = new TextEncoder();

function run(prog: Uint8Array, input: Uint8Array): { out: Uint8Array; exit: number } {
  const n = prog.length;
  const D = new Int32Array(256);
  const C = new Int32Array(64);
  const M = new Int32Array(256);
  const out = new Uint8Array(65536);
  let sp = 0, csp = 0, olen = 0, inPos = 0, pc = 0, steps = 0;
  const end = (exit: number) => ({ out: out.slice(0, olen), exit });
  for (;;) {
    if (steps === 1000000) return end(10); // step limit
    if (pc < 0 || pc >= n) return end(9); // pc outside program
    const op = prog[pc];
    if (!KNOWN[op]) return end(5); // unknown opcode
    const imm = IMM[op];
    if (pc + imm >= n) return end(9); // immediate cut off
    steps++;
    const next = pc + 1 + imm;
    if (sp < POPS[op]) return end(2); // stack underflow
    if (sp - POPS[op] + PUSHES[op] > 256) return end(3); // stack overflow
    pc = next;
    switch (op) {
      case 0x00: return end(0); // HALT
      case 0x01: D[sp++] = (prog[next - 1] << 24) >> 24; break; // PUSH8, sign-extended
      case 0x02: // PUSH32, little-endian
        D[sp++] = prog[next - 4] | (prog[next - 3] << 8) | (prog[next - 2] << 16) | (prog[next - 1] << 24);
        break;
      case 0x03: sp--; break; // POP
      case 0x04: D[sp] = D[sp - 1]; sp++; break; // DUP
      case 0x05: { const t = D[sp - 1]; D[sp - 1] = D[sp - 2]; D[sp - 2] = t; break; } // SWAP
      case 0x06: D[sp] = D[sp - 2]; sp++; break; // OVER
      case 0x07: { // ROT: a b c -> b c a
        const a = D[sp - 3]; D[sp - 3] = D[sp - 2];
        D[sp - 2] = D[sp - 1]; D[sp - 1] = a; break;
      }
      case 0x17: D[sp - 1] = -D[sp - 1] | 0; break; // NEG, wraps
      case 0x1b: D[sp - 1] = ~D[sp - 1]; break; // NOT
      case 0x10: case 0x11: case 0x12: case 0x13: case 0x14: case 0x15:
      case 0x16: case 0x18: case 0x19: case 0x1a: case 0x1c: case 0x1d:
      case 0x1e: case 0x20: case 0x21: case 0x22: {
        const b = D[sp - 1], a = D[sp - 2];
        let r = 0;
        if (op === 0x10) r = (a + b) | 0;
        else if (op === 0x11) r = (a - b) | 0;
        else if (op === 0x12) r = Math.imul(a, b); // low 32 bits of exact product
        else if (op === 0x13) { // DIV, truncated toward zero
          if (b === 0) return end(4);
          r = a === -2147483648 && b === -1 ? a : (a / b) | 0;
        } else if (op === 0x14) { // MOD, sign of a
          if (b === 0) return end(4);
          r = b === -1 ? 0 : (a % b) | 0;
        } else if (op === 0x15) { // DIVU, unsigned
          if (b === 0) return end(4);
          r = Math.floor((a >>> 0) / (b >>> 0)) | 0;
        } else if (op === 0x16) { // MODU, unsigned
          if (b === 0) return end(4);
          r = (a >>> 0) % (b >>> 0) | 0;
        } else if (op === 0x18) r = a & b;
        else if (op === 0x19) r = a | b;
        else if (op === 0x1a) r = a ^ b;
        else if (op === 0x1c) r = a << (b & 31); // only low 5 bits of shift count
        else if (op === 0x1d) r = a >> (b & 31); // arithmetic shift
        else if (op === 0x1e) r = (a >>> (b & 31)) | 0; // logical shift
        else if (op === 0x20) r = a === b ? 1 : 0;
        else if (op === 0x21) r = a < b ? 1 : 0;
        else r = (a >>> 0) < (b >>> 0) ? 1 : 0; // LTU, unsigned
        sp -= 2; D[sp++] = r; break;
      }
      case 0x30: case 0x31: case 0x32: { // JMP / JZ / JNZ, signed 16-bit offset
        const rel = ((prog[next - 2] | (prog[next - 1] << 8)) << 16) >> 16;
        let take = true;
        if (op !== 0x30) { const v = D[--sp]; take = op === 0x31 ? v === 0 : v !== 0; }
        if (take) pc = next + rel; // target unchecked: traps 9 at next step if outside
        break;
      }
      case 0x33: // CALL, unsigned 16-bit target
        if (csp === 64) return end(7);
        C[csp++] = next;
        pc = prog[next - 2] | (prog[next - 1] << 8);
        break;
      case 0x34: // RET
        if (csp === 0) return end(8);
        pc = C[--csp];
        break;
      case 0x40: { // LOAD
        const addr = D[sp - 1];
        if (addr < 0 || addr >= 256) return end(6);
        D[sp - 1] = M[addr]; break;
      }
      case 0x41: { // STORE: ( v addr -- )
        const addr = D[sp - 1], v = D[sp - 2];
        if (addr < 0 || addr >= 256) return end(6);
        sp -= 2; M[addr] = v; break;
      }
      case 0x50: { // PRINT: decimal ASCII, no newline
        const bytes = enc.encode(String(D[sp - 1]));
        if (olen + bytes.length > 65536) return end(11); // output limit, writes nothing
        sp--; out.set(bytes, olen); olen += bytes.length; break;
      }
      case 0x51: // PUTC: one byte, v AND 255
        if (olen + 1 > 65536) return end(11);
        out[olen++] = D[--sp] & 255; break;
      case 0x52: D[sp++] = inPos < input.length ? input[inPos++] : -1; break; // IN
      case 0x53: return end(D[--sp] & 255); // EXIT: low byte
    }
  }
}

for (const id of readdirSync(IN_DIR).sort()) {
  const prog = readFileSync(join(IN_DIR, id, 'program.bin'));
  const input = readFileSync(join(IN_DIR, id, 'input.bin'));
  const { out, exit } = run(prog, input);
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'stdout.bin'), out);
  writeFileSync(join(OUT_DIR, id, 'exit.txt'), String(exit) + '\n');
}
