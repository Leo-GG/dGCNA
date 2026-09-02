################################################################################
#Funtions for differential network analysis using scRNAseq and snRNAseq data
#Author: José A. Martínez-López
#Last modification: January 2022
################################################################################

#order: Positive or Negative. Order by Positive or negative diff. correlated number of links
GetHubGenes<-function(Connections,MinNeg=10,MinPos=10,both=TRUE,order="Positive"){
  
  Cpos<-Connections
  Cpos[Connections<=0]<-0
  
  Cneg<-Connections
  Cneg[Connections>=0]<-0
  
  Graph_Cpos<-graph_from_adjacency_matrix(data.matrix(abs(Cpos)),mode = "upper",diag = FALSE,weighted = TRUE)
  deg_Cpos<-degree(Graph_Cpos)
  
  Graph_Cneg<-graph_from_adjacency_matrix(data.matrix(abs(Cneg)),mode = "upper",diag = FALSE,weighted = TRUE)
  deg_Cneg<-degree(Graph_Cneg)
  
  NumberOfConnections<-data.frame(deg_Cpos,deg_Cneg)
  
  if (both){
    
    topgenes<-NumberOfConnections[NumberOfConnections[,"deg_Cpos"]>=MinPos & NumberOfConnections[,"deg_Cneg"]>=MinNeg,]
  
    }else{
      
    topgenes<-NumberOfConnections[NumberOfConnections[,"deg_Cpos"]>=MinPos | NumberOfConnections[,"deg_Cneg"]>=MinNeg,]
    
    }
  
  if (order=="Positive"){
    
    topgenes<-topgenes[order(topgenes[,"deg_Cpos"],decreasing = TRUE),]
  
  }else if(order=="Negative"){
    
    topgenes<-topgenes[order(topgenes[,"deg_Cneg"],decreasing = TRUE),]
    
  }
  return(topgenes)
}

GetTopHubGenes<-function(Connections,NUMBER_GENES){
  Cpos<-Connections
  Cpos[Connections<=0]<-0
  
  Cneg<-Connections
  Cneg[Connections>=0]<-0
  
  Graph_Cpos<-graph_from_adjacency_matrix(data.matrix(abs(Cpos)),mode = "upper",diag = FALSE,weighted = TRUE)
  deg_Cpos<-degree(Graph_Cpos)
  
  Graph_Cneg<-graph_from_adjacency_matrix(data.matrix(abs(Cneg)),mode = "upper",diag = FALSE,weighted = TRUE)
  deg_Cneg<-degree(Graph_Cneg)
  
  NumberOfConnections<-data.frame(deg_Cpos,deg_Cneg)

  x<-apply(NumberOfConnections,1,max)
  
  x_ordered<-x[order(x,decreasing = TRUE)]
  
  return(names(x_ordered[1:NUMBER_GENES]))
  
}

GetEigenCentralityScore<-function(Connections,MinNeg=0.1,MinPos=0.1,both=TRUE,order="Positive"){
  
  Cpos<-Connections
  Cpos[Connections<=0]<-0
  
  Cneg<-Connections
  Cneg[Connections>=0]<-0
  
  Graph_Cpos<-graph_from_adjacency_matrix(data.matrix(abs(Cpos)),mode = "upper",diag = FALSE,weighted = TRUE)
  ec_Cpos<-eigen_centrality(Graph_Cpos,directed = FALSE)
  
  Graph_Cneg<-graph_from_adjacency_matrix(data.matrix(abs(Cneg)),mode = "upper",diag = FALSE,weighted = TRUE)
  ec_Cneg<-eigen_centrality(Graph_Cneg,directed = FALSE)
  
  RankScores<-data.frame(ec_Cpos$vector,ec_Cneg$vector)
  
  if (both){
    
    topgenes<-RankScores[RankScores[,1]>=MinPos & RankScores[,2]>=MinNeg,]
    
  }else{
    
    topgenes<-RankScores[RankScores[,1]>=MinPos | RankScores[,2]>=MinNeg,]
    
  }
  
  if (order=="Positive"){
    
    topgenes<-topgenes[order(topgenes[,1],decreasing = TRUE),]
    
  }else if(order=="Negative"){
    
    topgenes<-topgenes[order(topgenes[,2],decreasing = TRUE),]
    
  }
  return(topgenes)
  
}

GetTopHubGenesByEigenCentrality<-function(Connections,TH=0.1){
  
  Graph<-graph_from_adjacency_matrix(data.matrix(Connections),mode = "upper",diag = FALSE,weighted = TRUE)
  ec<-eigen_centrality(Graph,directed = FALSE)
  
  RankScores<-data.frame(ec$vector)
  
  topgenes<-RankScores[RankScores[,1]>=TH,]
  
  topgenes<-topgenes[order(topgenes,decreasing = TRUE),]
  
  return(topgenes)
  
}


NetworkPruning<-function(network,weight_th){
  
  
  Graph<-graph_from_adjacency_matrix(data.matrix(abs(network)),mode = "upper",diag = FALSE,weighted = TRUE)
  
  deg<-degree(Graph)
  
  print(paste0("Average degree before pruning:",mean(deg)))
  
  #pruning
  network[network<weight_th & network>weight_th*(-1)]<-0
  
  Graph<-graph_from_adjacency_matrix(data.matrix(abs(network)),mode = "upper",diag = FALSE,weighted = TRUE)
  
  deg<-degree(Graph)
  
  print(paste0("Average degree after pruning:",mean(deg)))
  
  return(network)
}

SelectPositiveWeights<-function(connections,threshold=0){
  
  Cpos<-connections
  Cpos[connections<=threshold]<-0
  
  return(Cpos)
}

SelectNegativeWeights<-function(connections,threshold=0){
  
  Cneg<-connections
  Cneg[connections>=threshold]<-0
  
  return(abs(Cneg))
}

PlotDegreeDistribution<-function(connections,id_plot="Degree Distribution",normalize=TRUE){
  
  Mt_graph_all<-graph_from_adjacency_matrix(data.matrix(abs(connections)),mode = "upper",diag = FALSE,weighted = TRUE)
  
  dd_all<-degree_distribution((Mt_graph_all))
  
  degree_all<-degree(Mt_graph_all)
  
  probability_all<-dd_all[-1]

  nonzero.position_all<-which(probability_all!=0)
  
  probability_all<-probability_all[nonzero.position_all]

  degree_all<-1:max(degree_all)

  degree_all<-degree_all[nonzero.position_all]

  if(normalize){
    
    dat_ddis_all<-data.frame(probability_all,degree_all/nrow(connections))
    colnames(dat_ddis_all)<-c("pb","deg")
    lab<-"Degree (k)/N"
  
  }else{
      
    dat_ddis_all<-data.frame(probability_all,degree_all)
    colnames(dat_ddis_all)<-c("pb","deg")
    lab<-"Degree (k)"
    
  }
  sizetext<-20
  
  p<-ggplot(dat_ddis_all,aes(x=deg,y = pb))+
    scale_x_continuous(trans = "log10") +
    scale_y_continuous(trans = "log10") +
    annotation_logticks()+
    xlab(lab) +
    ylab("P(k)") +
    ggtitle(id_plot)+
    theme_light()+
    theme(text = element_text(size=sizetext),
          axis.text=element_text(size=sizetext)
          #axis.title.x=element_blank(),
          #axis.title.y=element_blank()
    )+
    geom_point()
  
  return(p)
  
}
