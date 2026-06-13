# DGN-A-network-traffic-prediction-framework-based-on-temporallylearnt-dynamic-graph-neural-networks
code about DGN
The spatial-temporal graph neural networks (STGNNs) based Internet network traffic forecasting (NTP) faces two challenges. First, the commonly used static network topology can’t really reflect the complicated and unknown network traffic dependencies. Second, Internet traffic is essentially irregular and nonstationary. This paper proposes a novel NTP model based on temporally-learnt dynamic graph neural networks, DGN. Specifically, our contributions are following. First, based on the initial embeddings of routers on static topology, the temporally changed relationship between routers, modelled as dynamic, weighted and directed graph, is learned with bilinear transform and nonlinear neural unit, which can fully characterize the irregular and nonstationary traffic fluctuation. Second, considering the asymmetric dependency between any pair of routers, in-link and out-link Graph Convolutional Networks (IO-GCNs) is designed to capture the dynamic spatial dependencies within a traffic matrix of each time step, and infer routers’ representations, which are further used as input to multiple layers of Gated Recurrent Units (GRU) to extract the temporal dependencies among traffic matrices. Experimental results on real network traffic data demonstrate that our proposed DGN outperforms other spatial-temporal GNN based and flow-based NTP schemes.

requirements:
python=3.8.0
tensorflow==2.6.0
torch==1.10.0+cu113
torchaudio==0.10.0+cu113
torchvision==0.11.1+cu113
torch-geometric==2.6.1
spektral==1.3.1
pandas==1.4.4
numpy==1.19.5
scikit-learn==1.3.2
