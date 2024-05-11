#### Environment

- Cluster with 3 nodes Nivdia T4 GPU

- Standard GKE cluster :region us-central1

- 10 GB ImageNet dataset dir: 

- ResNet 50 pre-trained model

  #### Cluster Creation

  Used the following gcloud command to create a cluster named 'my-dask':

  `gcloud container clusters create dask-cluster --num-nodes=3 --zone=us-central1 --disk-type=pd-standard --disk-size=10`

![image-20240510113853067](/Users/gongyitong/Library/Application Support/typora-user-images/image-20240510113853067.png)



#### Helm Installation of Dask

Installed Dask using Helm with the following command:

```
helm install my-dask dask/dask \
    --set scheduler.replicas=1 \
    --set worker.replicas=2
```



#### Check Cluster Status

Checked the cluster status using:

`kubectl get pods`

![image-20240510194617051](/Users/gongyitong/Library/Application Support/typora-user-images/image-20240510194617051.png)

To learn more about the release, try:



#### Monitoring Cluster Status via External Interface

Accessed the cluster status through:

http://10.244.0.6:8786/status

![image-20240510194738493](/Users/gongyitong/Library/Application Support/typora-user-images/image-20240510194738493.png)



#### Create Task Environment and Run Image

Built the Docker image with the following command:docker build -t image-processing 

![image-20240510193843370](/Users/gongyitong/Library/Application Support/typora-user-images/image-20240510193843370.png)

Image name: image-processing

Applied changes to the environment image to: `my-dask-scheduler.yaml` and `my-dask-worker.yaml`



#### Run Script in Environment

Connected to the scheduler using the Dask client with the following line of code:

```
client = Client('10.244.0.6:8786')  # Dask scheduler address and port
```

![image-20240510222546552](/Users/gongyitong/Library/Application Support/typora-user-images/image-20240510222546552.png)
