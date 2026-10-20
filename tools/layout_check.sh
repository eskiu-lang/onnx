#!/bin/sh
# Compares src/raw.esk with the C compiler's view of onnxruntime_c_api.h: every struct's
# size and field offsets, every OrtApi slot number and every enum value.
# Usage: tools/layout_check.sh INCLUDE_DIR [LIB_DIR]
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
inc=$1
[ -f "$inc/onnxruntime/onnxruntime_c_api.h" ] && inc=$inc/onnxruntime
lib=${2:-$inc/../lib}
cc=${CC:-cc}
eskiuc=${ESKIUC:-eskiuc}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
"$cc" -I"$inc" "$root/tools/layout/layout.c" -o "$tmp/c"
sed 's/^import <onnx\/raw>;//' "$root/tools/layout/layout.esk" > "$tmp/layout.esk"
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*)
    # The program needs onnxruntime.dll beside it: Windows looks in System32,
    # which may hold an older copy, before PATH.
    cp "$lib"/*.dll "$tmp/"
    "$eskiuc" "$root/src/raw.esk" "$tmp/layout.esk" -o "$tmp/e" -L"$lib"
    ;;
*)
    "$eskiuc" "$root/src/raw.esk" "$tmp/layout.esk" -o "$tmp/e" -L"$lib" --link-arg=-Wl,-rpath,"$lib"
    ;;
esac
"$tmp/c" > "$tmp/c.txt"
"$tmp/e" > "$tmp/e.txt"
diff "$tmp/c.txt" "$tmp/e.txt"
echo "layout ok: $(wc -l < "$tmp/c.txt" | tr -d ' ') sizes, offsets, slots and enum values"
