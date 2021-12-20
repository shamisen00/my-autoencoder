# my-autoencoder

## under construction
---
### **for development enviroment**

just open the container in vscode and 
```
python autoencoder.py --multirun setting=condition1,condition2
```

---
### **for production enviroment**
docker build -t "imagename" .

docker run --ipc=host --gpus all -it --name "containername" "imagename" bash

python autoencoder.py --multirun setting=condition1,condition2
