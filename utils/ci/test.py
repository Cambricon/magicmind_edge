import caffe
 
net = caffe.Net('resnet18_62000.prototxt')
 
for layer_name, blob in net.blobs.items():
    print("layer name: " + layer_name + ", data shape:" + str(blob.data.shape))


curl --output artifacts.zip --header "PRIVATE-TOKEN: 8taN7yiLy3ezymqsRyRg" "http://gitlab.software.cambricon.com/api/v4/projects/4261/jobs/42/artifacts"
