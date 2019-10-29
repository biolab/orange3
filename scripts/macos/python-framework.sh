#!/usr/bin/env bash

usage() {
    echo "$0 [ --version VERSION ] FRAMEWORKPATH

Fetch, extract and layout a macOS relocatable Python framework at FRAMEWORKPATH

Options:
    --version VERSION     Python version (default ${VERSION})
    --macos MACOSVER      Minimum supported macOS version (as of 3.6.5 and
                          3.7.0 the python.org provides binaries for 10.6
                          and 10.9 macOS versions; default ${MACOSVER})
    --install-certifi     If present then certifi pypi package will be
                          installed and its cert store linked in
                          \${PREFIX}/etc/openssl
    -v --verbose          Increase verbosity level

Note:
    Python >= 3.6 comes with a bundled openssl library build that is
    configured to load certificates from
    /Library/Frameworks/Python.frameworks/\${pyver}/etc/openssl

    This script will patch python's stdlib ssl.py to add a
    \${PREFIX}/etc/openssl/cert.pem (where \${PREFIX} is the runtime prefix)
    certificate store to the default verification chain. However it will only
    supply the file if the --install-certifi parameter is passed

Example
-------
    $ python-framework.sh ./
    $ Python.framework/Versions/3.5/bin/python3.5 --version
    $ python-framework.sh --version 2.7.12 ./2.7
    $ ./2.7/Python.framework/Versions/2.7/bin/python2.7 --version
"
}


VERSION=3.7.5
MACOSVER=10.9
VERBOSE_LEVEL=0
INSTALL_CERTIFI=


verbose() {
    local level=${1:?}
    shift 1
    if [[ ${VERBOSE_LEVEL:-0} -ge ${level} ]]; then
        echo "$@"
    fi
}

python-framework-fetch-pkg() {
    local cachedir=${1:?}
    local version=${2:?}
    local macosver=${3:-10.6}
    local versiondir=${version%%[abrpc]*}  # strip alpha, beta, rc component
    local filename=python-${version}-macosx${macosver}.pkg
    local url="https://www.python.org/ftp/python/${versiondir}/${filename}"
    mkdir -p "${cachedir}"
    if [[ -f "${cachedir}/${filename}" ]]; then
        verbose 1 "python-${version}-macosx{macosver}.pkg is present in cache"
        return 0
    fi
    local tmpfile=$(mktemp "${cachedir}/${filename}"-XXXX)
    cleanup-on-exit() {
        if [ -f "${tmpfile}" ]; then
            rm "${tmpfile}"
        fi
    }

    (
        trap cleanup-on-exit EXIT
        verbose 1 "Fetching ${url}"
        curl -sSL --fail -o "${tmpfile}" "${url}"
        mv "${tmpfile}" "${cachedir}/${filename}"
    )
}


python-framework-extract-pkg() {
    local targetdir=${1:?}
    local pkgpath=${2:?}
    local pkgfilename
    pkgfilename=$(basename "${pkgpath}")
    mkdir -p "${targetdir}"/Python.framework
    verbose 1 "Extracting framework at ${targetdir}/Python.framework"
    (
        tmpdir=$(mktemp -d -t python-framework-extract-pkg)
        cleanup-on-exit() {
            if [ -d "${tmpdir:?}" ] ; then
              rm -rf "${tmpdir:?}"
            fi
        }
        trap cleanup-on-exit EXIT
        pkgutil --expand "${pkgpath}" "${tmpdir:?}/${pkgfilename}" || exit 1
        tar -C "${targetdir}"/Python.framework \
            -xf "${tmpdir}/${pkgfilename}/Python_Framework.pkg/Payload" || exit 1
    )
}


python-framework-relocate() {
    local fmkdir=${1:?}
    if [[ ! ${fmkdir} =~ .*/Python.framework ]]; then
        echo "${fmkdir} is not a Python.framework" >&2
        return 1
    fi

    shopt -s nullglob
    local versions=( "${fmkdir}"/Versions/?.? )
    shopt -u nullglob

    if [[ ! ${#versions[*]} == 1 ]]; then
        echo "Single version framework expected (found: ${versions[@]})"
        return 2
    fi
    local ver_short=${versions##*/}
    local prefix="${fmkdir}"/Versions/${ver_short}
    local bindir="${prefix}"/bin
    local libdir="${prefix}"/lib

    local existingid=$(otool -X -D "${prefix}"/Python | tail -n 1)
    local anchor="${existingid%%/Python.framework*}"

    if [[ ! ${anchor} =~ /.* ]]; then
        echo "${anchor} is not an absolute path" 2>&1
        return 2
    fi

    chmod +w "${fmkdir}"/Versions/${ver_short}/Python
    # change main lib's install id
    install_name_tool \
        -id @rpath/Python.framework/Versions/${ver_short}/Python \
        "${fmkdir}"/Versions/${ver_short}/Python

    # Add the containing frameworks path to rpath
    install_name_tool \
        -add_rpath @loader_path/../../../ \
        "${fmkdir}"/Versions/${ver_short}/Python

    # all expected executable binaries
    local binnames=( python${ver_short}  python${ver_short}-32 \
                     pythonw${ver_short} pythonw${ver_short}-32 \
                     python${ver_short}m )

    for binname in "${binnames[@]}";
    do
        if [ -f "${bindir}/${binname}" ]; then
            install_name_tool \
                -change "${anchor}"/Python.framework/Versions/${ver_short}/Python \
                        "@executable_path/../Python" \
                "${bindir}/${binname}"
        fi
    done

    install_name_tool \
        -change "${anchor}"/Python.framework/Versions/${ver_short}/Python \
                "@executable_path/../../../../Python" \
        "${prefix}"/Resources/Python.app/Contents/MacOS/Python

    for lib in libncursesw libmenuw libformw libpanelw libssl libcrypto;
    do
        local libpath="${libdir}"/${lib}.dylib
        if [[ -f "${libpath}" ]]; then
            local libid=$(otool -X -D "${libpath}")
            local libbasename=$(basename "${libid}")

            chmod +w "${libpath}"
            install_name_tool \
                -id @rpath/Python.framework/Versions/${ver_short}/lib/${libbasename} \
                "$libpath"

            local librefs=( $(otool -X -L "${libpath}" | cut -d " " -f 1) )
            for libref in "${librefs[@]}"
            do
                if [[ ${libref} =~ ${anchor}/Python.framework/.* ]]; then
                    local libbasename=$(basename "${libref}")
                    install_name_tool \
                        -change "${libref}" @loader_path/${libbasename} \
                        "${libpath}"

                fi
            done
        fi
    done

    local dylibdir="${libdir}"/python${ver_short}/lib-dynload
    # _curses.so, _curses_panel.so, readline.so, ...
    local solibs=( "${dylibdir}"/*.so )
    for libpath in "${solibs[@]}"
    do
        local librefs=( $(otool -X -L "${libpath}" | cut -d " " -f 1) )
        for libref in "${librefs[@]}"
        do
            local strip_prefix="${anchor}"/Python.framework
            local strip_prefixn=$(( ${#strip_prefix} + 1 ))
            if [[ ${libref} =~ ${strip_prefix}/.* ]]; then
                local relpath=$(echo "${libref}" | cut -c $(( ${strip_prefixn} + 1))- )
                # TODO: should @loader_path be preferred here?
                install_name_tool \
                    -change "${libref}" \
                            @rpath/Python.framework/"${relpath}" \
                    "${libpath}"
            fi
        done
    done

    # files/modules which reference /Library/Frameworks/Python.framework/
    # - bin/*
    # - lib/pkgconfig/*.pc
    # - lib/python3.5/_sysconfigdata.py
    # - lib/python3.5/config-3.5m/python-config.py

    sed -i.bck s@prefix=${anchor}'.*'@prefix=\${pcfiledir}/../..@ \
        "${libdir}"/pkgconfig/python-${ver_short}.pc

    # 3.6.* has bundled libssl with a hardcoded absolute openssl_{cafile,capath}
    # (need to set SSL_CERT_FILE environment var in all scripts?
    # or patch ssl.py module to add certs to default verify list?)
    if [[ ${ver_short#*.} -ge 6 ]]; then
        patch-ssl "${prefix}"
    fi
}


# patch python 3.6 to add etc/openssl/cert.pem cert store located relative
# to the runtime prefix.
patch-ssl() {
    local prefix=${1:?}
    # lib/python relative to prefix
    local pylibdir=$(
        cd "${prefix}";
        shopt -s failglob;
        local path=( lib/python?.? )
        echo "${path:?}"
    )

    patch "${prefix}/${pylibdir}"/ssl.py - <<EOF
--- a/ssl.py    2017-04-07 10:26:34.000000000 +0200
+++ b/ssl.py    2017-04-07 10:52:59.000000000 +0200
@@ -448,6 +448,14 @@
         if sys.platform == "win32":
             for storename in self._windows_cert_stores:
                 self._load_windows_store_certs(storename, purpose)
+        # patched by python-framework.sh relocation script.
+        if sys.platform == "darwin":
+            cert_file = "../../etc/openssl/cert.pem"
+            path = os.path.join(os.path.dirname(__file__), cert_file)
+            try:
+                self.load_verify_locations(path)
+            except OSError:
+                pass
         self.set_default_verify_paths()

     @property
EOF
}


install-certifi() {
    local prefix=${1:?}
    "${prefix}"/bin/python?.? -B -m ensurepip
    "${prefix}"/bin/python?.? -B -m pip --isolated install certifi
    (
        mkdir -p "${prefix}"/etc/openssl
        cd "${prefix}"/etc/openssl
        ln -shf ../../lib/python?.?/site-packages/certifi/cacert.pem ./cert.pem
    )
    test -r "${prefix}"/etc/openssl/cert.pem
}


while [[ "${1:0:1}" == "-" ]]; do
    case "${1}" in
        --version)
            VERSION=${2:?"--version: missing argument"}
            shift 2;;
        --version=*)
            VERSION=${1##*=}
            shift 1;;
        --macos)
            MACOSVER=${2:?"--macos: missing argument"}
            shift 2;;
        --macos=*)
            MACOSVER=${1##*=}
            shift 1;;
        -v|--verbose)
            VERBOSE_LEVEL=$(( $VERBOSE_LEVEL + 1 ))
            shift 1;;
        --install-certifi)
            INSTALL_CERTIFI=1
            shift 1;;
        --help|-h)
            usage; exit 0;;
        -*)
            echo "Unrecognized argument ${1}" >&2
            usage >&2; exit 1;;
    esac
done

python-framework-fetch-pkg ~/.cache/pkgs/ ${VERSION} ${MACOSVER}
python-framework-extract-pkg \
    "${1:?"FRAMEWORKPATH argument is missing"}" \
    ~/.cache/pkgs/python-${VERSION}-macosx${MACOSVER}.pkg

python-framework-relocate "${1:?}"/Python.framework

# Update the Versions/Current symlink
(
    cd "${1:?}"/Python.framework/Versions
    shopt -s failglob
    ln -shf ?.? ./Current  # assuming single version framework
)

if [[ ${INSTALL_CERTIFI} ]]; then
    verbose 1 "Installing and linking certifi pypi package"
    install-certifi "${1:?}"/Python.framework/Versions/Current
fi
