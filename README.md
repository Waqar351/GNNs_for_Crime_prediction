# Modeling and Predicting Crimes in the City of São Paulo Using Graph Neural Networks (BRACIS 2024)

This repository contains the code and data for our paper titled *"Modeling and Predicting Crimes in the City of São Paulo Using Graph Neural Networks,"* which has been submitted to the BRACIS 2024 conference. The project is organized into three main components:

## 1. Crime Data Processing Pipeline (`crime-data`)
The `crime-data` folder includes a robust pipeline designed to integrate crime data from São Paulo’s Department of Public Safety with urban infrastructure into a graph derived from a street map. This integration provides a detailed and dynamic representation of the crime landscape in São Paulo. The pipeline is versatile and can be applied to various domains and geographical areas, provided the data is geolocated.

## 2. Dynamic Self-Attention Network (DySAT) (`dysat`)
The `dysat` folder implements the Dynamic Self-Attention Network (DySAT), an unsupervised graph embedding model that learns latent node representations to capture dynamic graph structures. DySAT employs self-attention mechanisms across both structural and temporal dimensions to generate accurate and robust node embeddings. The model's architecture includes:

- **Spatial Self-Attention:** Captures structural associations in each snapshot of the dynamic graph.
- **Temporal Self-Attention:** Learns temporal dependencies to understand how node representations evolve over time.
- **Output Layer:** Integrates spatial and temporal embeddings for tasks like node classification, link prediction, and anomaly detection.

## 3. Evolving Graph Convolutional Networks (EvolveGCN) (`evolvegcn`)
The `evolvegcn` folder contains the implementation of Evolving Graph Convolutional Networks (EvolveGCN), which adapt Graph Convolutional Networks (GCNs) to handle continuously evolving graphs. EvolveGCN uses time-series models, specifically LSTM networks, to predict and adjust GCN parameters over time, ensuring the network adapts to the graph's changing dynamics. This approach is essential for real-world applications where relationships between entities evolve, such as in social networks or communication networks.

## Usage
Each folder contains detailed instructions on setting up the environment, preparing the data, and running the models. Please refer to the respective `README` files within each folder for more specific guidelines.

## Citation
If you use this code, please cite our paper:
```
@INPROCEEDINGS{243068,
  AUTHOR="Waqar Hassan and Marvin Cabral and Thiago Ramos and Antonio Castelo Filho and Luis Gustavo Nonato",
  TITLE="Modeling and Predicting Crimes in the City of São Paulo Using Graph Neural Networks",
  BOOKTITLE="BRACIS 2024 () ",
  ADDRESS="",
  DAYS="23-21",
  MONTH="may",
  YEAR="2024",
  ABSTRACT="Crime prediction is a critical research area for enhancing public safety and optimizing law enforcement resource allocation, and machine learning techniques have had a significant impact in this field. Traditional machine learning models have long struggled to capture complex crime patterns, primarily due to the intricate interdependence of spatial and temporal data. However, recent advancements in machine learning, particularly with Graph Neural Networks (GNNs), offer a new perspective. GNNs have demonstrated remarkable success in various applications and they can also play a significant role in crime analysis and prediction. Therefore, in this work, we explore such a potential by examining two distinct spatiotemporal GNN architectures, namely Dynamic Self-Attention Network (DySAT) and Evolving Graph Convolutional Network (EvolveGCN), assessing and comparing their effectiveness for crime prediction. Moreover, we propose a data modeling framework that integrates crime, street map graphs, and urban data, which is fundamental to properly train the GNN models. As far as we know, there is no consolidated methodology to integrate those three modalities of data, being a relevant contribution of this work. Our findings underscore the effectiveness of GNNs in crime prediction tasks, offering valuable insights for researchers and practitioners in the field of crime prevention and public safety enhancement.",
  KEYWORDS="- Neural Networks; - Deep Learning; - Machine Learning and Data Mining; - Forecasting",
  URL="http://XXXXX/243068.pdf"
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
