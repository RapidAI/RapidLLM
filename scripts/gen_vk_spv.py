#!/usr/bin/env python3
"""Emit include/rapidllm/backend/vulkan_spv.h — no glslang required."""
from __future__ import annotations

import struct
from pathlib import Path

MAGIC = 0x07230203
VERSION = 0x00010000

OpCapability = 17
OpMemoryModel = 14
OpEntryPoint = 15
OpExecutionMode = 16
OpDecorate = 71
OpMemberDecorate = 72
OpTypeVoid = 19
OpTypeBool = 20
OpTypeInt = 21
OpTypeFloat = 22
OpTypeVector = 23
OpTypeArray = 28
OpTypeRuntimeArray = 29
OpTypeStruct = 30
OpTypePointer = 32
OpTypeFunction = 33
OpConstant = 43
OpFunction = 54
OpFunctionEnd = 56
OpVariable = 59
OpLoad = 61
OpStore = 62
OpAccessChain = 65
OpIAdd = 128
OpIMul = 132
OpFAdd = 129
OpFMul = 133
OpFDiv = 136
OpSLessThan = 177
OpIEqual = 170
OpConvertSToF = 111
OpExtInstImport = 11
OpExtInst = 12
OpLabel = 248
OpLoopMerge = 246
OpSelectionMerge = 247
OpBranch = 249
OpBranchConditional = 250
OpReturn = 253
OpPhi = 245


class Spv:
    def __init__(self):
        self.words: list[int] = []
        self.next = 1

    def id(self) -> int:
        i = self.next
        self.next += 1
        return i

    def emit(self, op: int, *ops: int) -> None:
        wc = 1 + len(ops)
        self.words.append((wc << 16) | op)
        self.words.extend(int(x) & 0xFFFFFFFF for x in ops)

    def emit_str(self, op: int, *pre: int, s: str = "") -> None:
        raw = s.encode("utf-8") + b"\x00"
        pad = (4 - (len(raw) % 4)) % 4
        raw += b"\x00" * pad
        extra = list(struct.unpack("<" + "I" * (len(raw) // 4), raw))
        ops = list(pre) + extra
        wc = 1 + len(ops)
        self.words.append((wc << 16) | op)
        self.words.extend(int(x) & 0xFFFFFFFF for x in ops)

    def finish(self) -> list[int]:
        return [MAGIC, VERSION, 0, self.next, 0] + self.words


def _const_u32(b: Spv, ty: int, val: int) -> int:
    i = b.id()
    b.emit(OpConstant, ty, i, val & 0xFFFFFFFF)
    return i


def _const_f32(b: Spv, ty: int, val: float) -> int:
    i = b.id()
    b.emit(OpConstant, ty, i, struct.unpack("<I", struct.pack("<f", val))[0])
    return i


def build_gemv() -> list[int]:
    """Y[row] = dot(W[row], X) for row in [0,m). local_size=1, dispatch m."""
    b = Spv()
    # ids
    t_void = b.id()
    t_fn = b.id()
    t_bool = b.id()
    t_u32 = b.id()
    t_i32 = b.id()
    t_f32 = b.id()
    t_v3u = b.id()
    t_ptr_in_v3u = b.id()
    t_rta = b.id()
    t_blk = b.id()
    t_push = b.id()
    t_ptr_unif_blk = b.id()
    t_ptr_pc = b.id()
    t_ptr_unif_f = b.id()
    t_ptr_pc_i = b.id()
    id_ext = b.id()
    id_main = b.id()
    id_gid = b.id()
    id_W = b.id()
    id_X = b.id()
    id_Y = b.id()
    id_P = b.id()

    b.emit(OpCapability, 1)  # Shader
    b.emit_str(OpExtInstImport, id_ext, s="GLSL.std.450")
    b.emit(OpMemoryModel, 0, 1)  # Logical GLSL450
    b.emit_str(OpEntryPoint, 5, id_main, id_gid, s="main")
    b.emit(OpExecutionMode, id_main, 17, 1, 1, 1)  # LocalSize 1 1 1

    # decorations
    b.emit(OpDecorate, id_gid, 11, 28)  # BuiltIn GlobalInvocationId
    b.emit(OpDecorate, t_rta, 5, 16)  # ArrayStride 16? NO — RuntimeArray of float stride 4
    # fix: ArrayStride 4
    b.words[-1] = 4
    b.emit(OpMemberDecorate, t_blk, 0, 35, 0)  # Offset 0
    b.emit(OpDecorate, t_blk, 3)  # BufferBlock
    b.emit(OpDecorate, id_W, 34, 0)  # DescriptorSet 0
    b.emit(OpDecorate, id_W, 33, 0)  # Binding 0
    b.emit(OpDecorate, id_X, 34, 0)
    b.emit(OpDecorate, id_X, 33, 1)
    b.emit(OpDecorate, id_Y, 34, 0)
    b.emit(OpDecorate, id_Y, 33, 2)
    b.emit(OpMemberDecorate, t_push, 0, 35, 0)
    b.emit(OpMemberDecorate, t_push, 1, 35, 4)
    b.emit(OpDecorate, t_push, 2)  # Block

    # types
    b.emit(OpTypeVoid, t_void)
    b.emit(OpTypeFunction, t_fn, t_void)
    b.emit(OpTypeBool, t_bool)
    b.emit(OpTypeInt, t_u32, 32, 0)
    b.emit(OpTypeInt, t_i32, 32, 1)
    b.emit(OpTypeFloat, t_f32, 32)
    b.emit(OpTypeVector, t_v3u, t_u32, 3)
    b.emit(OpTypePointer, t_ptr_in_v3u, 1, t_v3u)  # Input
    b.emit(OpTypeRuntimeArray, t_rta, t_f32)
    b.emit(OpTypeStruct, t_blk, t_rta)
    b.emit(OpTypeStruct, t_push, t_i32, t_i32)
    b.emit(OpTypePointer, t_ptr_unif_blk, 2, t_blk)  # Uniform
    b.emit(OpTypePointer, t_ptr_pc, 9, t_push)  # PushConstant=9
    b.emit(OpTypePointer, t_ptr_unif_f, 2, t_f32)
    b.emit(OpTypePointer, t_ptr_pc_i, 9, t_i32)

    c0 = _const_u32(b, t_i32, 0)
    c1 = _const_u32(b, t_i32, 1)
    cf0 = _const_f32(b, t_f32, 0.0)

    b.emit(OpVariable, t_ptr_in_v3u, id_gid, 1)
    b.emit(OpVariable, t_ptr_unif_blk, id_W, 2)
    b.emit(OpVariable, t_ptr_unif_blk, id_X, 2)
    b.emit(OpVariable, t_ptr_unif_blk, id_Y, 2)
    b.emit(OpVariable, t_ptr_pc, id_P, 9)

    # function
    b.emit(OpFunction, t_void, id_main, 0, t_fn)
    lab_entry = b.id()
    b.emit(OpLabel, lab_entry)

    gid_v = b.id()
    b.emit(OpLoad, t_v3u, gid_v, id_gid)
    row_u = b.id()
    b.emit(OpCompositeExtract, t_u32, row_u, gid_v, 0)
    row = b.id()
    b.emit(OpBitcast := 124, t_i32, row, row_u)  # Bitcast u->i is OK for low ids

    # load m, n
    p_m_ptr = b.id()
    b.emit(OpAccessChain, t_ptr_pc_i, p_m_ptr, id_P, c0)
    m = b.id()
    b.emit(OpLoad, t_i32, m, p_m_ptr)
    p_n_ptr = b.id()
    b.emit(OpAccessChain, t_ptr_pc_i, p_n_ptr, id_P, c1)
    n = b.id()
    b.emit(OpLoad, t_i32, n, p_n_ptr)

    # if (row >= m) return
    ge = b.id()
    b.emit(OpSLessThan, t_bool, ge, row, m)  # row < m
    lab_body = b.id()
    lab_end = b.id()
    b.emit(OpSelectionMerge, lab_end, 0)
    b.emit(OpBranchConditional, ge, lab_body, lab_end)

    b.emit(OpLabel, lab_body)
    # loop j
    lab_hdr = b.id()
    lab_cont = b.id()
    lab_merge = b.id()
    lab_do = b.id()
    b.emit(OpBranch, lab_hdr)
    b.emit(OpLabel, lab_hdr)
    j = b.id()
    acc = b.id()
    b.emit(OpPhi, t_i32, j, c0, lab_body, None)  # placeholder, fix after
    # We'll emit Phi properly: OpPhi type id {val, block}*
    # Remove the broken emit
    b.words.pop()
    b.words.pop()
    # reconstruct Phi after we know lab_cont
    # Actually emit Phi now with body + later patch? Easier: use structured loop with known labels.

    # Restart loop section: I already emitted Branch and Label hdr.
    # Emit Phi for j and acc.
    b.emit(OpPhi, t_i32, j, c0, lab_body, c0, lab_cont)  # j from cont will be patched — need jnext
    # This is getting messy. Use a cleaner loop pattern.

    raise SystemExit("use v2 builder")


def c_array(name: str, words: list[int]) -> str:
    lines = [f"static const uint32_t {name}[] = {{"]
    row: list[str] = []
    for w in words:
        row.append(f"0x{w:08x}u")
        if len(row) == 6:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    return "\n".join(lines)


# --- v2: write words with an explicit instruction list so Phis are easy ---


class Asm:
    def __init__(self):
        self.ids = {}
        self.next = 1
        self.body: list[list[int | str]] = []

    def i(self, name: str | None = None) -> int:
        if name and name in self.ids:
            return self.ids[name]
        n = self.next
        self.next += 1
        if name:
            self.ids[name] = n
        return n

    def op(self, opcode: int, *args: int | str) -> None:
        self.body.append([opcode, *args])

    def op_str(self, opcode: int, *args: int | str, s: str) -> None:
        raw = s.encode("utf-8") + b"\x00"
        raw += b"\x00" * ((4 - (len(raw) % 4)) % 4)
        extra = list(struct.unpack("<" + "I" * (len(raw) // 4), raw))
        self.body.append([opcode, *args, *extra])

    def op_entry(self, model: int, fn: str, name: str, *iface: str) -> None:
        raw = name.encode("utf-8") + b"\x00"
        raw += b"\x00" * ((4 - (len(raw) % 4)) % 4)
        extra = list(struct.unpack("<" + "I" * (len(raw) // 4), raw))
        self.body.append([OpEntryPoint, model, fn, *extra, *iface])

    def finish(self) -> list[int]:
        words: list[int] = []
        for item in self.body:
            opcode = int(item[0])
            ops = []
            for x in item[1:]:
                if isinstance(x, str):
                    ops.append(self.ids[x])
                else:
                    ops.append(int(x) & 0xFFFFFFFF)
            wc = 1 + len(ops)
            words.append((wc << 16) | opcode)
            words.extend(ops)
        return [MAGIC, VERSION, 0, self.next, 0] + words


def gemv_spv() -> list[int]:
    a = Asm()
    # pre-allocate names used in decorations / entry
    for n in (
        "tvoid", "tfn", "tbool", "tu32", "ti32", "tf32", "tv3u", "pin_v3u",
        "trta", "tblk", "tpush", "pub", "ppc", "puf", "ppi",
        "ext", "main", "gid", "W", "X", "Y", "P",
        "c0", "c1", "cf0",
        "entry", "body", "end", "hdr", "doit", "cont", "merge",
        "gidv", "rowu", "row", "pm", "m", "pn", "n", "ok",
        "j", "acc", "jlt", "base", "idx", "wp", "wv", "xp", "xv", "prod", "acc2",
        "j1", "yp",
    ):
        a.i(n)

    a.op(OpCapability, 1)
    a.op_str(OpExtInstImport, "ext", s="GLSL.std.450")
    a.op(OpMemoryModel, 0, 1)
    a.op_entry(5, "main", "main", "gid")
    a.op(OpExecutionMode, "main", 17, 1, 1, 1)

    a.op(OpDecorate, "gid", 11, 28)
    a.op(OpDecorate, "trta", 6, 4)  # ArrayStride 4
    a.op(OpMemberDecorate, "tblk", 0, 35, 0)
    a.op(OpDecorate, "tblk", 3)  # BufferBlock
    a.op(OpDecorate, "W", 34, 0)
    a.op(OpDecorate, "W", 33, 0)
    a.op(OpDecorate, "X", 34, 0)
    a.op(OpDecorate, "X", 33, 1)
    a.op(OpDecorate, "Y", 34, 0)
    a.op(OpDecorate, "Y", 33, 2)
    a.op(OpMemberDecorate, "tpush", 0, 35, 0)
    a.op(OpMemberDecorate, "tpush", 1, 35, 4)
    a.op(OpDecorate, "tpush", 2)

    a.op(OpTypeVoid, "tvoid")
    a.op(OpTypeFunction, "tfn", "tvoid")
    a.op(OpTypeBool, "tbool")
    a.op(OpTypeInt, "tu32", 32, 0)
    a.op(OpTypeInt, "ti32", 32, 1)
    a.op(OpTypeFloat, "tf32", 32)
    a.op(OpTypeVector, "tv3u", "tu32", 3)
    a.op(OpTypePointer, "pin_v3u", 1, "tv3u")
    a.op(OpTypeRuntimeArray, "trta", "tf32")
    a.op(OpTypeStruct, "tblk", "trta")
    a.op(OpTypeStruct, "tpush", "ti32", "ti32")
    a.op(OpTypePointer, "pub", 2, "tblk")
    a.op(OpTypePointer, "ppc", 9, "tpush")
    a.op(OpTypePointer, "puf", 2, "tf32")
    a.op(OpTypePointer, "ppi", 9, "ti32")

    a.op(OpConstant, "ti32", "c0", 0)
    a.op(OpConstant, "ti32", "c1", 1)
    a.op(OpConstant, "tf32", "cf0", struct.unpack("<I", struct.pack("<f", 0.0))[0])

    a.op(OpVariable, "pin_v3u", "gid", 1)
    a.op(OpVariable, "pub", "W", 2)
    a.op(OpVariable, "pub", "X", 2)
    a.op(OpVariable, "pub", "Y", 2)
    a.op(OpVariable, "ppc", "P", 9)

    a.op(OpFunction, "tvoid", "main", 0, "tfn")
    a.op(OpLabel, "entry")
    a.op(OpLoad, "tv3u", "gidv", "gid")
    a.op(OpCompositeExtract := 81, "tu32", "rowu", "gidv", 0)
    a.op(124, "ti32", "row", "rowu")  # Bitcast
    a.op(OpAccessChain, "ppi", "pm", "P", "c0")
    a.op(OpLoad, "ti32", "m", "pm")
    a.op(OpAccessChain, "ppi", "pn", "P", "c1")
    a.op(OpLoad, "ti32", "n", "pn")
    a.op(OpSLessThan, "tbool", "ok", "row", "m")
    a.op(OpSelectionMerge, "end", 0)
    a.op(OpBranchConditional, "ok", "body", "end")

    a.op(OpLabel, "body")
    a.op(OpBranch, "hdr")

    a.op(OpLabel, "hdr")
    a.op(OpPhi, "ti32", "j", "c0", "body", "j1", "cont")
    a.op(OpPhi, "tf32", "acc", "cf0", "body", "acc2", "cont")
    a.op(OpSLessThan, "tbool", "jlt", "j", "n")
    a.op(OpLoopMerge, "merge", "cont", 0)
    a.op(OpBranchConditional, "jlt", "doit", "merge")

    a.op(OpLabel, "doit")
    a.op(OpIMul, "ti32", "base", "row", "n")
    a.op(OpIAdd, "ti32", "idx", "base", "j")
    a.op(OpAccessChain, "puf", "wp", "W", "c0", "idx")
    a.op(OpLoad, "tf32", "wv", "wp")
    a.op(OpAccessChain, "puf", "xp", "X", "c0", "j")
    a.op(OpLoad, "tf32", "xv", "xp")
    a.op(OpFMul, "tf32", "prod", "wv", "xv")
    a.op(OpFAdd, "tf32", "acc2", "acc", "prod")
    a.op(OpIAdd, "ti32", "j1", "j", "c1")
    a.op(OpBranch, "cont")

    a.op(OpLabel, "cont")
    a.op(OpBranch, "hdr")

    a.op(OpLabel, "merge")
    a.op(OpAccessChain, "puf", "yp", "Y", "c0", "row")
    a.op(OpStore, "yp", "acc")
    a.op(OpBranch, "end")

    a.op(OpLabel, "end")
    a.op(OpReturn)
    a.op(OpFunctionEnd)
    return a.finish()


def rms_spv() -> list[int]:
    """Invocation 0: y = rmsnorm(x, gamma, eps) with +1. Dispatch 1."""
    a = Asm()
    for n in (
        "tvoid", "tfn", "tbool", "tu32", "ti32", "tf32", "tv3u", "pin_v3u",
        "trta", "tblk", "tpush", "pub", "ppc", "puf", "ppi", "ppf",
        "ext", "main", "gid", "X", "G", "Y", "P",
        "c0", "c1", "cf0", "cf1",
        "entry", "okb", "end",
        "h1", "d1", "c1l", "m1",
        "h2", "d2", "c2l", "m2",
        "gidv", "ixu", "ix", "ok",
        "pn", "n", "pe", "eps",
        "i", "ss", "ilt", "xp", "xv", "sq", "ss2", "i1",
        "nf", "mean", "den", "inv",
        "j", "jlt", "xj", "gj", "gp", "yp", "t", "ymid", "ypv", "j1",
    ):
        a.i(n)

    a.op(OpCapability, 1)
    a.op_str(OpExtInstImport, "ext", s="GLSL.std.450")
    a.op(OpMemoryModel, 0, 1)
    a.op_entry(5, "main", "main", "gid")
    a.op(OpExecutionMode, "main", 17, 1, 1, 1)

    a.op(OpDecorate, "gid", 11, 28)
    a.op(OpDecorate, "trta", 6, 4)
    a.op(OpMemberDecorate, "tblk", 0, 35, 0)
    a.op(OpDecorate, "tblk", 3)
    a.op(OpDecorate, "X", 34, 0)
    a.op(OpDecorate, "X", 33, 0)
    a.op(OpDecorate, "G", 34, 0)
    a.op(OpDecorate, "G", 33, 1)
    a.op(OpDecorate, "Y", 34, 0)
    a.op(OpDecorate, "Y", 33, 2)
    a.op(OpMemberDecorate, "tpush", 0, 35, 0)
    a.op(OpMemberDecorate, "tpush", 1, 35, 4)
    a.op(OpDecorate, "tpush", 2)

    a.op(OpTypeVoid, "tvoid")
    a.op(OpTypeFunction, "tfn", "tvoid")
    a.op(OpTypeBool, "tbool")
    a.op(OpTypeInt, "tu32", 32, 0)
    a.op(OpTypeInt, "ti32", 32, 1)
    a.op(OpTypeFloat, "tf32", 32)
    a.op(OpTypeVector, "tv3u", "tu32", 3)
    a.op(OpTypePointer, "pin_v3u", 1, "tv3u")
    a.op(OpTypeRuntimeArray, "trta", "tf32")
    a.op(OpTypeStruct, "tblk", "trta")
    a.op(OpTypeStruct, "tpush", "ti32", "tf32")
    a.op(OpTypePointer, "pub", 2, "tblk")
    a.op(OpTypePointer, "ppc", 9, "tpush")
    a.op(OpTypePointer, "puf", 2, "tf32")
    a.op(OpTypePointer, "ppi", 9, "ti32")
    a.op(OpTypePointer, "ppf", 9, "tf32")

    a.op(OpConstant, "ti32", "c0", 0)
    a.op(OpConstant, "ti32", "c1", 1)
    a.op(OpConstant, "tf32", "cf0", struct.unpack("<I", struct.pack("<f", 0.0))[0])
    a.op(OpConstant, "tf32", "cf1", struct.unpack("<I", struct.pack("<f", 1.0))[0])

    a.op(OpVariable, "pin_v3u", "gid", 1)
    a.op(OpVariable, "pub", "X", 2)
    a.op(OpVariable, "pub", "G", 2)
    a.op(OpVariable, "pub", "Y", 2)
    a.op(OpVariable, "ppc", "P", 9)

    a.op(OpFunction, "tvoid", "main", 0, "tfn")
    a.op(OpLabel, "entry")
    a.op(OpLoad, "tv3u", "gidv", "gid")
    a.op(81, "tu32", "ixu", "gidv", 0)
    a.op(124, "ti32", "ix", "ixu")
    a.op(OpIEqual, "tbool", "ok", "ix", "c0")
    a.op(OpSelectionMerge, "end", 0)
    a.op(OpBranchConditional, "ok", "okb", "end")

    a.op(OpLabel, "okb")
    a.op(OpAccessChain, "ppi", "pn", "P", "c0")
    a.op(OpLoad, "ti32", "n", "pn")
    a.op(OpAccessChain, "ppf", "pe", "P", "c1")
    a.op(OpLoad, "tf32", "eps", "pe")

    # ss loop
    a.op(OpBranch, "h1")
    a.op(OpLabel, "h1")
    a.op(OpPhi, "ti32", "i", "c0", "okb", "i1", "c1l")
    a.op(OpPhi, "tf32", "ss", "cf0", "okb", "ss2", "c1l")
    a.op(OpSLessThan, "tbool", "ilt", "i", "n")
    a.op(OpLoopMerge, "m1", "c1l", 0)
    a.op(OpBranchConditional, "ilt", "d1", "m1")
    a.op(OpLabel, "d1")
    a.op(OpAccessChain, "puf", "xp", "X", "c0", "i")
    a.op(OpLoad, "tf32", "xv", "xp")
    a.op(OpFMul, "tf32", "sq", "xv", "xv")
    a.op(OpFAdd, "tf32", "ss2", "ss", "sq")
    a.op(OpIAdd, "ti32", "i1", "i", "c1")
    a.op(OpBranch, "c1l")
    a.op(OpLabel, "c1l")
    a.op(OpBranch, "h1")
    a.op(OpLabel, "m1")
    a.op(OpConvertSToF, "tf32", "nf", "n")
    a.op(OpFDiv, "tf32", "mean", "ss", "nf")
    a.op(OpFAdd, "tf32", "den", "mean", "eps")
    # InverseSqrt = GLSL.std.450 opcode 32
    a.op(OpExtInst, "tf32", "inv", "ext", 32, "den")

    a.op(OpBranch, "h2")
    a.op(OpLabel, "h2")
    a.op(OpPhi, "ti32", "j", "c0", "m1", "j1", "c2l")
    a.op(OpSLessThan, "tbool", "jlt", "j", "n")
    a.op(OpLoopMerge, "m2", "c2l", 0)
    a.op(OpBranchConditional, "jlt", "d2", "m2")
    a.op(OpLabel, "d2")
    a.op(OpAccessChain, "puf", "xp", "X", "c0", "j")
    a.op(OpLoad, "tf32", "xj", "xp")
    a.op(OpAccessChain, "puf", "gp", "G", "c0", "j")
    a.op(OpLoad, "tf32", "gj", "gp")
    a.op(OpFAdd, "tf32", "t", "cf1", "gj")
    a.op(OpFMul, "tf32", "ymid", "xj", "inv")
    a.op(OpFMul, "tf32", "ypv", "ymid", "t")
    a.op(OpAccessChain, "puf", "yp", "Y", "c0", "j")
    a.op(OpStore, "yp", "ypv")
    a.op(OpIAdd, "ti32", "j1", "j", "c1")
    a.op(OpBranch, "c2l")
    a.op(OpLabel, "c2l")
    a.op(OpBranch, "h2")
    a.op(OpLabel, "m2")
    a.op(OpBranch, "end")

    a.op(OpLabel, "end")
    a.op(OpReturn)
    a.op(OpFunctionEnd)
    return a.finish()


def main() -> None:
    # Fix rms double-ypv: regenerate rms with unique id
    out = Path(__file__).resolve().parents[1] / "include" / "rapidllm" / "backend" / "vulkan_spv.h"
    g = gemv_spv()
    r = rms_spv()
    text = (
        "#pragma once\n#include <stdint.h>\n\n"
        "// Generated by scripts/gen_vk_spv.py — BufferBlock SSBO + push constants.\n"
        f"{c_array('kSpvGemvF32', g)}\n\n"
        f"{c_array('kSpvRmsNorm', r)}\n"
    )
    out.write_text(text, encoding="utf-8")
    print("wrote", out, "gemv_words", len(g), "rms_words", len(r))


if __name__ == "__main__":
    main()
