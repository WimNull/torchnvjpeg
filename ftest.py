import os, sys

def test1():
    # pyd所在路径
    sys.path.append('build/lib.linux-x86_64-3.8/torchnvjpeg')

    module_name = "lib"
    exec("import %s" % module_name)
    from pybind11_stubgen import ModuleStubsGenerator


    module = ModuleStubsGenerator(module_name)
    module.parse()
    module.write_setup_py = False

    with open("%s.pyi" % module_name, "w") as fp:
        fp.write("#\n# Automatically generated file, do not edit!\n#\n\n")
        fp.write("\n".join(module.to_lines()))

def test2():
    import torchnvjpeg
    print(dir(torchnvjpeg))
    nvjpg = torchnvjpeg.NvJpeg()
    imp = '/data/source/torchnvjpeg/sample/cat.jpg'
    with open(imp, 'rb') as fin:
        data = fin.read()
    img = nvjpg.decode(data)
    print(img.shape, img.device)
    endata = nvjpg.encode(img)
    print(len(endata))
    with open('out.jpg', 'wb') as fout:
        fout.write(endata)



if __name__=='__main__':
    funname = 'test1'
    if len(sys.argv)>=2:
        funname = sys.argv[1]
        sys.argv.remove(funname)
    globals()[funname]()

