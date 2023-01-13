import os

PROJ_ROOT_PATH = os.environ.get("PROJ_ROOT_PATH")

filename_path=str(PROJ_ROOT_PATH+ '/data/output/infer_cpp_output_qint8_mixed_float16_1')

filenames = [os.path.join(filename_path, file) for file in os.listdir(filename_path)]

for f in filenames:
    # print("filename ",f)
    with open(f, 'r+') as file:
        a = file.read()
        a =a.strip('[').strip(']\n').replace(';', ' ')
        # print(a)
        file_new = open(f,"w")
        file_new.write(a)
        file_new.close()
print("Finish gen predlist ")
