#!/usr/bin/env python3
"""Generate src/raw.esk (and the layout check programs) from onnxruntime_c_api.h.

Usage: tools/gen.py INCLUDE_DIR
INCLUDE_DIR holds onnxruntime_c_api.h and onnxruntime_error_code.h, as the ONNX Runtime
release archives and Homebrew ship them (include/ or include/onnxruntime/).

The C API is one exported function, OrtGetApiBase, and a table of function pointers
(struct OrtApi) called by position. Eskiu calls a fn value as code(env, args...), so each
wrapper here builds a fn value whose code is the table entry and whose env is the C
function's first argument. That needs the first argument to travel in an integer register:
a pointer, an integer or an enum. Functions whose first argument is a float or a struct
passed by value are left out and listed at the end of src/raw.esk.
"""
import os
import re
import sys

KEYWORDS = set("""let int float double bool char string void struct packed union interface enum fn
operator asm volatile static escaping must_use async await const sizeof free_closure thread_create
thread_join for in while do if else switch match case default break return import extern intrinsic
alloc_with null true false try catch finally throw defer errdefer continue int8 int16 int32 int64 uint
uint8 uint16 uint32 uint64 type deprecated dbg""".split())

SCALARS = {
    "uint8_t": "uint8", "uint16_t": "uint16", "uint32_t": "uint32", "uint64_t": "uint64",
    "int8_t": "int8", "int16_t": "int16", "int32_t": "int32", "int64_t": "int64",
    "size_t": "uint64", "int": "int", "float": "float", "double": "double", "bool": "bool",
    "char": "char", "unsigned": "uint32",
}

NARROW = {"bool": "int", "char": "int", "int8": "int", "uint8": "uint32", "int16": "int", "uint16": "uint32"}

API_STRUCTS = ("OrtApi", "OrtModelEditorApi", "OrtCompileApi", "OrtInteropApi", "OrtModelPackageApi")


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def drop_cplusplus(s):
    """Drops the `#ifdef __cplusplus` branches (keeping their #else part)."""
    out, stack = [], []
    for line in s.split("\n"):
        d = line.strip()
        if d.startswith("#if"):
            stack.append("cpp" if d.split()[:2] == ["#ifdef", "__cplusplus"] else "other")
            if stack[-1] == "cpp":
                continue
        elif d.startswith("#else") and stack and stack[-1].startswith("cpp"):
            stack[-1] = "cpp-else"
            continue
        elif d.startswith("#endif") and stack:
            if stack.pop().startswith("cpp"):
                continue
        if "cpp" in stack:
            continue
        out.append(line)
    return "\n".join(out)


def strip_annotations(s):
    s = re.sub(r"\b_[A-Za-z][A-Za-z0-9_]*_\s*\((?:[^()]|\([^()]*\))*\)", " ", s)
    s = re.sub(r"\b_[A-Za-z][A-Za-z0-9_]*_\b", " ", s)
    for w in ("NO_EXCEPTION", "ORT_ALL_ARGS_NONNULL", "ORT_MUST_USE_RESULT", "ORT_API_CALL"):
        s = s.replace(w, " ")
    return s


def ident(name):
    return name + "_" if name in KEYWORDS else name


def split_top(s, sep):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
        if ch == sep and depth != 0:
            pass
    if cur.strip():
        out.append(cur)
    return out


def body_at(s, start):
    depth, i = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        i += 1
    return s[start:i - 1], i


class Gen:
    def __init__(self):
        self.opaque = []
        self.enums = []
        self.enum_names = set()
        self.fnptrs = []
        self.structs = []
        self.struct_names = set()
        self.slots = []
        self.funcs = []
        self.skipped = []
        self.unknown = set()

    def ctype(self, t, param=False):
        """Maps a C type to Eskiu. Returns None for what a wrapper cannot pass."""
        t = " ".join(t.replace("struct ", " ").replace("enum ", " ").split())
        stars = t.count("*")
        words = [w for w in t.replace("*", " ").split() if w not in ("const", "volatile")]
        if words[:2] == ["unsigned", "char"]:
            words = ["uint8_t"]
        elif words[:2] == ["signed", "char"]:
            words = ["int8_t"]
        elif words[:2] == ["unsigned", "int"]:
            words = ["unsigned"]
        base = words[0] if words else "void"
        if base == "char" and stars >= 1:
            return "*" * (stars - 1) + "string"
        if base == "void":
            return "*" * stars + "void" if stars else "void"
        if base in ("OrtStatusPtr",):
            return "*" * stars + "*OrtStatus"
        if base in SCALARS:
            name = SCALARS[base]
            if stars == 0 and param and name in NARROW:
                name = NARROW[name]
            if stars == 0 and not param and name == "char":
                name = "int8"
            return "*" * stars + name
        if base == "ORTCHAR_T":
            return "*" * stars + "ORTCHAR_T"
        if base in self.enum_names:
            return "*" * stars + base
        if base in self.fnptr_names or base in self.struct_names or base in self.opaque_names:
            if stars == 0 and base in self.struct_names:
                return None
            return "*" * stars + base
        self.unknown.add(base)
        return None

    def collect_names(self, s):
        self.opaque_names = set()
        for m in re.finditer(r"ORT_RUNTIME_CLASS\((\w+)\);", s):
            self.opaque_names.add("Ort" + m.group(1))
        for m in re.finditer(r"typedef\s+struct\s+(\w+)\s+\w+\s*;", s):
            self.opaque_names.add(m.group(1))
        self.fnptr_names = set()
        for m in re.finditer(r"typedef\s+[\w\s\*]+?\(\s*(?:ORT_API_CALL\s*)?\*\s*(\w+)\s*\)\s*\(", s):
            self.fnptr_names.add(m.group(1))
        self.fnptr_names.add("OrtCustomThreadHandle")
        self.fnptr_names.add("OrtStatusPtr")
        for m in re.finditer(r"(?:typedef\s+)?enum\s+(\w+)\s*\{", s):
            self.enum_names.add(m.group(1))

    def parse_enums(self, s):
        for m in re.finditer(r"(?:typedef\s+)?enum\s+(\w+)\s*\{", s):
            body, _ = body_at(s, m.end())
            members = []
            for item in body.split(","):
                item = " ".join(item.split())
                if item:
                    k, _, v = item.partition("=")
                    members.append((k.strip(), v.strip()))
            if m.group(1) not in [e[0] for e in self.enums]:
                self.enums.append((m.group(1), members))

    def fields(self, name, body):
        out = []
        for decl in split_top(strip_annotations(body), ";"):
            decl = " ".join(decl.split())
            if not decl:
                continue
            fp = re.match(r"(.*?)\(\s*\*\s*(\w+)\s*\)\s*\(.*\)$", decl, re.S)
            fp = fp or re.match(r"(ORT_API2_STATUS)\((\w+)\s*,.*\)$", decl, re.S)
            if fp:
                out.append(("*void", fp.group(2), fp.group(2)))
                continue
            fm = re.match(r"(.*?)(\w+)\s*(\[\s*(\w+)\s*\])?$", decl)
            t = self.ctype(fm.group(1))
            if t is None:
                raise SystemExit("struct %s: cannot map field '%s'" % (name, decl))
            if fm.group(3):
                t = "%s[%s]" % (t, fm.group(4))
            out.append((t, ident(fm.group(2)), fm.group(2)))
        return out

    def parse_structs(self, s):
        for m in re.finditer(r"(typedef\s+(?:const\s+)?)?struct\s+(\w+)\s*\{", s):
            name = m.group(2)
            if name in API_STRUCTS:
                continue
            body, end = body_at(s, m.end())
            self.struct_names.add(name)
            self.structs.append((name, body))
        self.opaque_names -= self.struct_names
        self.opaque_names |= set(API_STRUCTS)
        self.opaque = sorted(self.opaque_names)

    def parse_api(self, s):
        i = s.index("struct OrtApi {") + len("struct OrtApi {")
        body, _ = body_at(s, i)
        members = [" ".join(x.split()) for x in split_top(body, ";")]
        members = [x for x in members if x]
        for slot, mem in enumerate(members):
            mem = mem.strip()
            m = re.match(r"ORT_API2_STATUS\((\w+)\s*,?(.*)\)$", mem)
            if m:
                ret, name, params = "OrtStatusPtr", m.group(1), m.group(2)
            else:
                m = re.match(r"ORT_CLASS_RELEASE\((\w+)\)$", mem)
                if m:
                    ret, name, params = "void", "Release" + m.group(1), "Ort%s* input" % m.group(1)
                else:
                    m = re.match(r"ORT_API_T\(((?:[^,()]|\([^()]*\))+),\s*(\w+)\s*,?(.*)\)$", mem)
                    if m:
                        ret, name, params = m.group(1), m.group(2), m.group(3)
                    else:
                        m = re.match(r"(.*?)\(\s*ORT_API_CALL\s*\*\s*(\w+)\s*\)\s*\((.*)\).*$", mem)
                        if not m:
                            raise SystemExit("cannot parse OrtApi member %d: %s" % (slot, mem))
                        ret, name, params = m.group(1), m.group(2), m.group(3)
            self.slots.append(name)
            self.add_func(slot, ret, name, params)

    def add_func(self, slot, ret, name, params):
        params = " ".join(strip_annotations(params).split())
        rt = self.ctype(strip_annotations(ret))
        if rt is None:
            self.skipped.append((name, "returns a struct by value"))
            return
        ps = []
        if params and params != "void":
            for i, p in enumerate(split_top(params, ",")):
                p = " ".join(p.split())
                fp = re.match(r"(.*?)\(\s*\*\s*(\w+)\s*\)\s*\(.*\)$", p)
                if fp:
                    ps.append(("*void", ident(fp.group(2))))
                    continue
                pm = re.match(r"(.*?)\s*(\w+)\s*(\[\s*\w*\s*\])?$", p)
                ptype, pname = pm.group(1), pm.group(2)
                if pm.group(3):
                    ptype += "*"
                if not ptype.strip() or ptype.strip() in ("const", "struct"):
                    ptype, pname = p, "a%d" % i
                t = self.ctype(ptype, param=True)
                if t is None:
                    self.skipped.append((name, "parameter '%s' is a struct passed by value" % p))
                    return
                ps.append((t, ident(pname)))
        if ps and ps[0][0] in ("float", "double"):
            self.skipped.append((name, "its first argument is a %s" % ps[0][0]))
            return
        if ret.strip() == "bool":
            rt = "uint8"
        self.funcs.append((slot, rt, name, ps))

    def parse(self, texts):
        s = drop_cplusplus(strip_comments("\n".join(texts)))
        self.collect_names(s)
        self.parse_enums(s)
        self.parse_structs(s)
        self.fnptrs = sorted(self.fnptr_names - {"OrtStatusPtr"})
        self.struct_fields = [(n, self.fields(n, b)) for n, b in self.structs]
        self.parse_api(s)
        if self.unknown:
            raise SystemExit("unknown C types: %s" % ", ".join(sorted(self.unknown)))

    def fn_type(self, rt, ps):
        args = []
        for t, _ in ps[1:]:
            args.append("int" if t in self.enum_names else t)
        r = "int" if rt in self.enum_names else rt
        return "fn(%s)->%s" % (", ".join(args), r)

    def env_expr(self, t, name):
        if t.startswith("*") or t == "string":
            return "(*void)%s" % name
        if t in ("uint64",):
            return "(*void)%s" % name
        if t in ("uint32", "uint16", "uint8"):
            return "(*void)(uint64)%s" % name
        return "(*void)(int64)%s" % name

    def emit(self, version, ort_version):
        out = ["// Generated by tools/gen.py from onnxruntime_c_api.h (ONNX Runtime %s). Do not edit." % ort_version,
               "// size_t is uint64: ONNX Runtime ships for 64-bit platforms only.",
               "//",
               "// ort_api() is the OrtApi table for ORT_API_VERSION, or for the loaded library's",
               "// older version down to ORT_API_MIN_VERSION. Each ort_Name(...) calls OrtApi::Name",
               "// through its slot.",
               "", '#if defined(__wasm__) || !(defined(__x86_64__) || defined(__aarch64__))',
               '#error "onnx: ONNX Runtime ships for 64-bit x86 and Arm only (macOS, Linux, Windows, iOS, Android)"',
               '#endif', "", '#pragma link("onnxruntime")', ""]
        out.append("const uint32 ORT_API_VERSION = %d;" % version)
        out.append("const uint32 ORT_API_MIN_VERSION = 16;   // the high-level API's slots exist since 1.16")
        out.append("const int ORT_API_SLOTS = %d;" % len(self.slots))
        out.append("")
        out.append("#ifdef _WIN32")
        out.append("type ORTCHAR_T = uint16;")
        out.append("#else")
        out.append("type ORTCHAR_T = uint8;")
        out.append("#endif")
        out.append("")
        for name in self.opaque:
            out.append("struct %s { int _opaque; }" % name)
        out.append("")
        for name in self.fnptrs:
            out.append("type %s = *void;" % name)
        out.append("")
        for name, members in self.enums:
            out.append("enum %s {" % name)
            out.append(",\n".join("    %s%s" % (k, " = " + v if v else "") for k, v in members))
            out.append("}")
            out.append("")
        for name, fields in self.struct_fields:
            out.append("struct %s {" % name)
            for t, f, _ in fields:
                out.append("    %s %s;" % (t, f))
            out.append("}")
            out.append("")
        out.append("extern *OrtApiBase OrtGetApiBase();")
        out.append("")
        out.append("struct _OrtRaw {")
        out.append("    *void code;")
        out.append("    *void env;")
        out.append("}")
        out.append("")
        sigs = {}
        for slot, rt, name, ps in self.funcs:
            ft = self.fn_type(rt, ps)
            if ft not in sigs:
                sigs[ft] = "_OrtF%d" % len(sigs)
                out.append("struct %s { %s f; }" % (sigs[ft], ft))
        out.append("")
        out.append("struct _OrtFGetApi { fn()->**void f; }")
        out.append("struct _OrtFGetVersion { fn()->string f; }")
        out.append("")
        out.append("**void _ort_api_table;")
        out.append("uint32 _ort_api_version;")
        out.append("")
        out.append("// The ONNX Runtime version string of the loaded library, such as \"%s\"." % ort_version)
        out.append("string ort_version_string() {")
        out.append("    *OrtApiBase base = OrtGetApiBase();")
        out.append("    _OrtRaw r = _OrtRaw{base.GetVersionString, null};")
        out.append("    fn()->string f = ((*_OrtFGetVersion)&r).f;")
        out.append("    return f();")
        out.append("}")
        out.append("")
        out.append("// OrtApiBase::GetApi(version): the function table, or null when unsupported.")
        out.append("**void ort_get_api(uint32 version) {")
        out.append("    *OrtApiBase base = OrtGetApiBase();")
        out.append("    _OrtRaw r = _OrtRaw{base.GetApi, (*void)(uint64)version};")
        out.append("    fn()->**void f = ((*_OrtFGetApi)&r).f;")
        out.append("    return f();")
        out.append("}")
        out.append("")
        out.append("// The OrtApi table, fetched once: ORT_API_VERSION, or the loaded library's own API")
        out.append("// version when it is older (ONNX Runtime 1.N provides version N), down to")
        out.append("// ORT_API_MIN_VERSION. A slot added after ort_api_version() must not be called.")
        out.append("// null if the library is older than that.")
        out.append("**void ort_api() {")
        out.append("    if (_ort_api_table != null) return _ort_api_table;")
        out.append("    string s = ort_version_string();")
        out.append("    uint32 minor = 0;")
        out.append("    int i = 0;")
        out.append("    while (s[i] != 0 && s[i] != '.') i++;")
        out.append("    if (s[i] == '.') i++;")
        out.append("    while (s[i] >= '0' && s[i] <= '9') {")
        out.append("        minor = minor * 10 + (uint32)((int)s[i] - 48);")
        out.append("        i++;")
        out.append("    }")
        out.append("    uint32 v = minor < ORT_API_VERSION ? minor : ORT_API_VERSION;")
        out.append("    if (v < ORT_API_MIN_VERSION) return null;")
        out.append("    _ort_api_table = ort_get_api(v);")
        out.append("    if (_ort_api_table != null) _ort_api_version = v;")
        out.append("    return _ort_api_table;")
        out.append("}")
        out.append("")
        out.append("// The OrtApi version ort_api() uses (0 before it is fetched or when unavailable).")
        out.append("uint32 ort_api_version() {")
        out.append("    ort_api();")
        out.append("    return _ort_api_version;")
        out.append("}")
        out.append("")
        for i, name in enumerate(self.slots):
            out.append("const int ORT_SLOT_%s = %d;" % (name, i))
        out.append("")
        for slot, rt, name, ps in self.funcs:
            ft = self.fn_type(rt, ps)
            sig = ", ".join("%s %s" % p for p in ps)
            ret = "bool" if rt == "uint8" and name == "TensorTypeAndShape_HasShape" else rt
            out.append("%s ort_%s(%s) {" % (ret, name, sig))
            env = self.env_expr(ps[0][0], ps[0][1]) if ps else "null"
            out.append("    _OrtRaw r = _OrtRaw{ort_api()[ORT_SLOT_%s], %s};" % (name, env))
            out.append("    %s f = ((*%s)&r).f;" % (ft, sigs[ft]))
            call = "f(%s)" % ", ".join(p[1] for p in ps[1:])
            if rt == "void":
                out.append("    %s;" % call)
            elif ret == "bool":
                out.append("    return %s != 0;" % call)
            else:
                out.append("    return %s;" % call)
            out.append("}")
            out.append("")
        if self.skipped:
            out.append("// Not wrapped (Eskiu passes the first argument in the fn value's env slot):")
            for name, why in self.skipped:
                out.append("//   %s: %s" % (name, why))
        return "\n".join(out).rstrip() + "\n"

    def emit_layout(self):
        c = ["#include <stdio.h>", "#include <stddef.h>", '#include "onnxruntime_c_api.h"', "int main(void) {"]
        e = ["import <io>;", "import <onnx/raw>;", "", "int main() {"]
        for name, fields in self.struct_fields:
            c.append('    printf("%s %%zu\\n", sizeof(struct %s));' % (name, name))
            e.append('    printf("%s %%d\\n", (int)sizeof(%s));' % (name, name))
            e.append("    { %s v;" % name)
            for t, f, cf in fields:
                c.append('    printf("%s.%s %%zu\\n", offsetof(struct %s, %s));' % (name, cf, name, cf))
                e.append('      printf("%s.%s %%d\\n", (int)((*uint8)&v.%s - (*uint8)&v));' % (name, cf, f))
            e.append("    }")
        c.append('    printf("OrtApi slots %zu\\n", sizeof(OrtApi) / sizeof(void*));')
        e.append('    printf("OrtApi slots %d\\n", ORT_API_SLOTS);')
        for name in self.slots:
            c.append('    printf("OrtApi.%s %%zu\\n", offsetof(OrtApi, %s) / sizeof(void*));' % (name, name))
            e.append('    printf("OrtApi.%s %%d\\n", ORT_SLOT_%s);' % (name, name))
        for name, members in self.enums:
            for k, _ in members:
                c.append('    printf("%s %%d\\n", (int)%s);' % (k, k))
                e.append('    printf("%s %%d\\n", (int)%s);' % (k, k))
        c += ["    return 0;", "}"]
        e += ["    return 0;", "}"]
        return "\n".join(c) + "\n", "\n".join(e) + "\n"


def main():
    inc = sys.argv[1]
    if os.path.exists(os.path.join(inc, "onnxruntime", "onnxruntime_c_api.h")):
        inc = os.path.join(inc, "onnxruntime")
    texts = []
    for name in ("onnxruntime_error_code.h", "onnxruntime_c_api.h"):
        with open(os.path.join(inc, name)) as f:
            texts.append(f.read())
    version = int(re.search(r"#define ORT_API_VERSION (\d+)", texts[1]).group(1))
    g = Gen()
    g.parse(texts)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "src", "raw.esk"), "w") as f:
        f.write(g.emit(version, "1.%d" % version))
    c, e = g.emit_layout()
    os.makedirs(os.path.join(root, "tools", "layout"), exist_ok=True)
    with open(os.path.join(root, "tools", "layout", "layout.c"), "w") as f:
        f.write(c)
    with open(os.path.join(root, "tools", "layout", "layout.esk"), "w") as f:
        f.write(e)
    print("slots %d wrapped %d skipped %d enums %d structs %d opaque %d" % (
        len(g.slots), len(g.funcs), len(g.skipped), len(g.enums), len(g.structs), len(g.opaque)))
    for name, why in g.skipped:
        print("  skipped %s: %s" % (name, why))


main()
