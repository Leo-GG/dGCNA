################################################################################
#Funtions for differential network analysis using scRNAseq and snRNAseq data
#Author: José A. Martínez-López
#Last modification: January 2022
################################################################################

#Return panel of samples containing cells of the selected community
#Samples: Panel list with samples data
#ClustersAnn: Factor with cluster annotation and CellID
#Selection: Character vector indicating cluster number or numbers
#min_cells: min number of cells per sample
#format: choose Matrix or dgCMatrix. Default dgCMatrix

SelectCellCommunities <- function(Samples,ClustersAnn,Selection,min_cells=1,format="dgCMatrix",verbose=TRUE){

  samplesComm<-list()

  x<-data.matrix(ClustersAnn)
  n<-row.names(x)
  n<-n[x %in% Selection]


  for (p in 1:length(Samples)) {

    s<-as.matrix(Samples[[p]])
    sel<-colnames(s) %in% n
    r<-s[,sel]

    if (sum(sel)>=min_cells){

      if(format=="dgCMatrix"){

          samplesComm[[names(Samples)[p]]]<-as(r,"dgCMatrix")

      } else if (format=="Matrix"){

          samplesComm[[names(Samples)[p]]]<-as.matrix(r)

      }

      if(verbose){
        print(paste0(sum(sel)," cells found for sample ",names(Samples)[p]))
      }
    }else{
      if(verbose){
        print(paste0("Low number of cells found for sample ",names(Samples)[p]))
      }
    }
  }
  return(samplesComm)
}

#Return a panel list with a fixed number of cells per sample
#Samples: Panel list with samples data
#NumberOfCells: Number of cells per sample

GetFixedNumberOfCells <- function(Samples,NumberOfCells=10){

  m<-NULL
  d<-sapply(Samples,dim)

  if(all(d[2,]>=NumberOfCells)){

    m<-lapply(Samples,SubsamplingCells,n=NumberOfCells,replacement=FALSE)

  } else{

    print(paste0("Error: some samples don't contain enought number of cells"))
  }

  return(m)
}


#Return list with sample states
#Samples: Panel list with samples data
#StatesAnn: Factor with states annotation (i.e.:Control-Disease) and CellID

SamplesStates <- function (Samples,StatesAnn){

  sampleslist<-list()
  x<-data.matrix(StatesAnn)
  s<-character(length = length(Samples))
  names(s)<-names(Samples)

  for (p in 1:length(Samples)) {
    y<-x[row.names(x) %in% colnames(Samples[[p]]),]
    s[p]<-names(table(y))
  }

  for (i in levels(StatesAnn)){
   sampleslist[[i]]<-names(s[s==i])
  }
  #for loop using lapply
  #lapply(levels(StatesAnn),function(i){sampleslist[[i]]<-names(s[s==i])})
  return(sampleslist)
}

#Return a dgCMatrix containing cells from selected samples
#Samples: Panel list with donors data
#Selection: character vector or list containing donors ID to merge. Exclude samples not indicated in this vector

CombineSamples <- function (Samples,Selection,verbose=TRUE){

  m<-NULL

  if(is.list(Selection)){
    Selection<-unlist(Selection)
  }

  s<-Selection %in% names(Samples)

  if(all(s)){

    m<-do.call(cbind,Samples[Selection])

    if(verbose){
      print(paste0("Dimensions of the matrix: ","Rows ",dim(m)[1]," Columns ",dim(m)[2]))
    }

  }else{

    print("Error: Some selected donors are not in the panel list")

  }

  return(m)
}

#Return a vector with Sample ID per cell
#Samples: Panel list with samples data

GetGroupIDperCell <- function(Samples){

  f<-list()

  s<-unlist(lapply(Samples,ncol))
  nam<-names(Samples)

  for (i in 1:length(s)){
    f[[i]]<-rep(nam[i],s[i])
  }
  return(unlist(f))
}


#Return a dgCMatrix containing subsampling of cells
#Samples: Panel list with samples data or dgCMatrix
#n: number of cells to take from Selection
#replacement: should sampling be with replacement

SubsamplingCells <- function (Samples,n,replacement=FALSE){

  x<-NULL

  if (is.list(Samples)){ #Subsampling from all donors

    m<-do.call(cbind,Samples)

  } else{ #Subsampling from one donor
    m<-Samples

  }

  if (n<=dim(m)[2]){
    r<-sample(1:dim(m)[2],size=n,replace=replacement)
    x<-m[,r]

  }else{
    print(paste0("Number of cells exceed matrix dimension"))
  }

  return(x)
}

#Return a character vector with random samples ID
#s: list or character vector with samples states
#n: number of random samples
SelectRandomSamples <- function(s,n,replacement=FALSE){

  x<-NULL
  if(is.list(s)){
    s<-unlist(s)
  }
  if(n<=length(s)){
    if(is.character(s)){
      r<-sample(1:length(s),size = n,replace=replacement)
      x<-s[r]
    }else{
      print("Error: provide a character vector or list with samples ID")
    }
  }else{
    print(paste0("Number of samples exceed population"))
  }
  return(x)
}

#Return markers of one community
#Samples: Panel list with samples data
#ClustersAnn: Factor with cluster annotation and CellID
#Community: Character indicating the community
#RestOfCommunities: Character vector of communities to compare
FindMarkers <- function (Samples,ClustersAnn,Community,RestOfCommunities,method="ExpressionRatio",min_expression=1){

  if (method=="ExpressionRatio"){

    Mcomm<-do.call(cbind,SelectCommunity(Samples,ClustersAnn,Community,verbose = FALSE))
    m1<-rowMeans(Mcomm)

    m<-list()
    for (i in RestOfCommunities){
      M<-do.call(cbind,SelectCommunity(Samples,ClustersAnn,i,verbose = FALSE))
      m[[i]]<-rowMeans(M)
    }
    m2<-do.call(cbind,m)
    m3<-rowSums(cbind(m1,m2))

    Specificity<-m1[m3>=min_expression]/m3[m3>=min_expression]
  }
  return(Specificity[order(Specificity,decreasing = TRUE)])
}

#Return a character vector containing genes of interest
#input_mat: dgCMatrix matrix with expression
#min_expr: minimum expression per cell
#cell_proportion: minimum proportion of cells with min_expr
#method: Proportion or NumberOfCells
GetExpressedGenes<-function(input_mat,min_expr=1,method="Proportion",min_cells=2,cell_proportion=0.3){

  if(method=="NumberOfCells"){ #min expression in at least min_cells

    m<-input_mat>=min_expr
    g<-row.names(input_mat[rowSums(m)>=min_cells,])

  }else if(method=="Proportion"){ #proportion of cells with min_expr

    prop<-rowSums(input_mat>=min_expr)/ncol(input_mat)
    g<-row.names(input_mat[prop>=cell_proportion,])
  }
  return(g)
}

#Return a character vector with common genes expressed in the samples indicated
#input_data: Panel list with samples data
#samp: character vector with sample names of interest
GetCommonExpressedGenes<-function(input_data,samp=NULL,min_expr=1,met="NumberOfCells",min_cells=2,cell_proportion=0.3,print=TRUE,FileName="BackgroundGeneSet.txt"){

  commongenes<-NULL

  if(is.list(input_data)){

    if(is.null(samp)){

      g<-lapply(input_data,GetExpressedGenes,min_expr=min_expr,method=met,min_cells=min_cells,cell_proportion=cell_proportion)

    } else{

      g<-lapply(input_data[samp],GetExpressedGenes,min_expr=min_expr,method=met,min_cells=min_cells,cell_proportion=cell_proportion)

    }

    commongenes<-Reduce(intersect,g)

  } else{

    print("Please provide a list with samples data")

  }

  if(print){
    write.table(commongenes,file=FileName,sep=";",quote = FALSE,col.names = FALSE,row.names=FALSE)
  }

  return(commongenes)
}

#Return multilevel linear models per gene and condition
#input_data: Panel list with samples data
#sample_groups: list with sample states
#genes: character vector containing genes
#model: lmer, blmer or glmer (NB)
#normalization: FALSE or TRUE (log2 normalization)

GetLinearModel<-function(input_data,sample_groups,genes=NULL,model="lmer",normalization=TRUE){

  x<-NULL

  if(is.null(genes)){

    print("Please provide a character vector with gene names")

  }

  else {

    if(normalization){
      dt<-lapply(input_data,function(x){ #filter genes and normalize
        y<-as.matrix(x[row.names(x) %in% genes,])
        return(log2(y+1))
      })
    }else{
      dt<-lapply(input_data,function(x){ #filter genes without normalization
        y<-as.matrix(x[row.names(x) %in% genes,])
        return(y)
      })
    }

    dt_groups<-lapply(sample_groups,function(x){
      y<-CombineSamples(dt,x,verbose=FALSE)
      return(y)
    })

    dr<-lapply(sample_groups,function(x){ #For donors
      y<-GetGroupIDperCell(input_data[x])
      return(y)
    })

    if (model=="lmer"){

      m<-mapply(function(u,v){

        ld<-apply(t(u),2,list)

        r<-lapply(ld,function(x){
          df<-data.frame(x,v,stringsAsFactors=FALSE)
          names(df)<-c("Expression","Donor")
          lm<-lmer(Expression ~ 1 + (1 | Donor),data=df)
          return(lm)
        })

        return(r)
      },dt_groups,dr,SIMPLIFY = FALSE)

    } else if (model=="glmer"){

      m<-mapply(function(u,v){

        ld<-apply(t(u),2,list)

        r<-sapply(ld,function(x){
          df<-data.frame(x,v,stringsAsFactors=FALSE)
          names(df)<-c("Expression","Donor")
          lm<-glmer.nb(Expression ~ 1 + (1 | Donor),data=df)
          return(lm)
        })

        return(r)
      },dt_groups,dr,SIMPLIFY = FALSE)

    } else if (model=="blmer"){

      m<-mapply(function(u,v){

        ld<-apply(t(u),2,list)

        r<-sapply(ld,function(x){
          df<-data.frame(x,v,stringsAsFactors=FALSE)
          names(df)<-c("Expression","Donor")
          lm<-blmer(Expression ~ 1 + (1 | Donor),data=df)
          return(lm)
        })

        return(r)
      },dt_groups,dr,SIMPLIFY = FALSE)

    }
  }

  return(m)

}
#Calculate similarities for non nested data
#Return a list of matrices (1 per group, example cotrol,disease...) containing similarities between genes (i.e. Pearson Correlation)
#input_data: Panel list with samples data
#sample_groups: list with sample states
#genes: character vector containing genes
#method: pearson or spearman Correlation Coefficient
#normalization: FALSE or TRUE (log2 normalization)

FindSimilaritiesNonNestedData<-function(input_data,sample_groups=NULL,genes=NULL,method="pearson",normalization=TRUE){

  x<-NULL

  if(is.null(genes)){

    print("Please provide a character vector with gene names")

  }

  else {

    if(normalization){
      dt<-lapply(input_data,function(x){ #filter genes and normalize
        y<-as.matrix(x[row.names(x) %in% genes,])
        return(log2(y+1))
      })

    }else{
      dt<-lapply(input_data,function(x){ #filter genes and normalize
        y<-as.matrix(x[row.names(x) %in% genes,])
        return(y)
      })
    }

    if(!is.null(sample_groups)){

      dt_groups<-lapply(sample_groups,function(x){
        y<-CombineSamples(dt,x)
        return(y)
      })
    }else{ #groups are already defined

      dt_groups<-dt

    }

    if (method=="pearson"){

      m<-lapply(dt_groups,t)
      x<-lapply(m,cor,method="pearson")

    } else if(method=="spearman"){

      m<-lapply(dt_groups,t)
      x<-lapply(m,cor,method="spearman")
    }
  }

  return(x)
}

#Calculate similarites for nested data using multilevel linear regression
#Return a list of matrices (1 per group, example cotrol,disease...) containing similarities between genes (i.e. Pearson Correlation)
#method: pearson or spearman Correlation Coefficient
#data_lm: data of multilevel linear models

FindSimilaritiesForNestedData <- function (data_lm,method="pearson"){


  res<-lapply(data_lm,function(x){
    v<-sapply(x,residuals)
    return(v)
  })

  if (method=="pearson"){

    x<-lapply(res,cor,method="pearson")

  } else if(method=="spearman"){

    x<-lapply(res,cor,method="spearman")

  }

  return(x)
}


#Boostrapping to get random similarity differences. Random selection of donors and considering multilevel regression to calculate
#similarities

GetSimilarityDifferencesFromRandomDonors<-function(input_data,sample_groups,genes=NULL,model="lmer",
                                normalization=TRUE,replacement=TRUE,method="pearson",T,Nctrl,Ncases,verbose=TRUE,ncores=1){

  d<-mclapply(1:T,function(i){

    #Get random groups of donors
    random_donors_g1<-SelectRandomSamples(sample_groups,Nctrl,replacement)
    random_donors_g2<-SelectRandomSamples(sample_groups,Ncases,replacement)


    random_groups<-list(g1=random_donors_g1,g2=random_donors_g2)

    if(verbose){
      print(paste0("Running iteration: ",i))
      print(paste0("Samples Group 1: ",random_groups[["g1"]]))
      print(paste0("Samples Group 2: ",random_groups[["g2"]]))
    }

    lm<-GetLinearModel(input_data,random_groups,genes,model,normalization)

  #Find similarities random samples
    sm_random<-FindSimilaritiesForNestedData(data_lm=lm,method)

    print(names(sm_random))

    diff<-sm_random[["g2"]]-sm_random[["g1"]]

    return(diff)
  },mc.cores=ncores)

  return(d)
}

GetMaskFromRandomDonors<-function(input_data,sample_groups,SimilarityDifferences,genes=NULL,model="lmer",
                                normalization=TRUE,replacement=TRUE,method="pearson",T,Nctrl,Ncases,verbose=TRUE,ncores=1){

  P<-matrix(0,nrow=nrow(SimilarityDifferences),ncol=ncol(SimilarityDifferences))

  Nsamples<-Nctrl+Ncases
  Nctrl_1<-Nctrl+1

  N<-ceiling(T/ncores)
  Treal<-N*ncores

  if(verbose){
    print(paste0("Number of iterations indicated: ",T,". Iterations to be run: ",Treal))
  }

  for (i in 1:N){

      d<-mclapply(1:ncores,function(x){

        #Get random groups of donors
        if(replacement){

            random_donors_g1<-SelectRandomSamples(sample_groups,Nctrl,replacement)
            random_donors_g2<-SelectRandomSamples(sample_groups,Ncases,replacement)

            random_groups<-list(g1=random_donors_g1,g2=random_donors_g2)

        }else{

            random_donors<-SelectRandomSamples(sample_groups,Nsamples,replacement)

            random_groups<-list(g1=random_donors[1:Nctrl],g2=random_donors[Nctrl_1:Nsamples])

        }

        if(verbose){
          print(paste0("Running round: ",i,". Core: ",x))
          print(paste0("Samples Group 1: ",random_groups[["g1"]]))
          print(paste0("Samples Group 2: ",random_groups[["g2"]]))
        }

        lm<-GetLinearModel(input_data,random_groups,genes,model,normalization)

      #Find similarities random samples
        sm_random<-FindSimilaritiesForNestedData(data_lm=lm,method)

        print(names(sm_random))

        random_diff<-sm_random[["g2"]]-sm_random[["g1"]]

        W<-CompareDifferences(SimDiff=SimilarityDifferences,RandomSimDiff=random_diff)

        print(paste0("dimensions SimilarityDifferences: ",dim(SimilarityDifferences)))
        print(paste0("dimensions RandomSimilarityDifferences: ",dim(random_diff)))
        print(paste0("dimensions comparison: ",dim(W)))
        print(paste0("Elements comparison: ",table(W)))
        print(paste0("class W: ",class(W)))

        return(W)

      },mc.cores=ncores)

      G<-Reduce('+',d)
      P<-P+G
  }

  return(P/Treal)
}

#Return a list of size T with matrices containing random similarity differences
#Boostrapping to get random similarities. Random selection of cells
#Nctrl: total number of cells to pick for group 1
#Ncases: total number of cells to pick for group 2
#T: Number of times

GetSimilarityDifferencesFromRandomCells<-function(input_data,sample_groups,genes=NULL,
                                                  normalization=TRUE,replacement=TRUE,method="pearson",Nctrl,Ncases,T,verbose=TRUE,ncores=1){

  allcells<-CombineSamples(input_data,unlist(sample_groups))

  d<-mclapply(1:T,function(i){

    if(verbose){
      print(paste0("Running iteration: ",i))
    }

    random_cells_g1<-SubsamplingCells(allcells,Nctrl,replacement)
    random_cells_g2<-SubsamplingCells(allcells,Ncases,replacement)

    #two random groups
    random_groups<-list(g1=random_cells_g1,g2=random_cells_g2)

    sm_random<-FindSimilaritiesNonNestedData(input_data=random_groups,sample_groups=NULL,genes,method,normalization)
    diff<-sm_random[["g2"]]-sm_random[["g1"]]
    return(diff)

  },mc.cores=ncores)

  return(d)
}

#Get confident intervals for the difference of two similarities
#Return a list of 2 matrices with confident intervals of the difference between two similarities, lower and upper bounds
#RandomSimDiff: list with matrices containing random similarity differences
#CI: Confident Interval. Default 0.95

GetConfidentIntervals<-function(RandomSimDiff,CIlower=0.95,CIupper=0.95){

  alpha_upper<-1-CIupper
  alpha_lower<-1-CIlower
  l<-alpha_lower/2
  u<-CIupper+alpha_upper/2

  low<-apply(simplify2array(RandomSimDiff),1:2,quantile,probs=l)
  up<-apply(simplify2array(RandomSimDiff),1:2,quantile,probs=u)

  return(list(lower=low,upper=up))
}

#Get Network Connections
#SimilarityDifferences: Matrix with Similarity Differences
#CISimilarityDifferences: Confident Intervals for differences obtained with GetConfidentIntervals functions

GetNetworkConnections<-function(SimilarityDifferences,CISimilarityDifferences){

  NegativeDiffLinks<-(SimilarityDifferences<CISimilarityDifferences[["lower"]])*(-1)
  PositiveDiffLinks<-(SimilarityDifferences>CISimilarityDifferences[["upper"]])*(1)

  print(paste0("Number of Negative Links Detected: ",nnzero(NegativeDiffLinks)))
  print(paste0("Number of Positive Links Detected: ",nnzero(PositiveDiffLinks)))

  return(NegativeDiffLinks+PositiveDiffLinks)
}




#Compare two matrices of Similarity Differences (two-conditions vs random)
#Return a binary matrix. TRUE are elements greater (positive elements) or lower (negative elements) than random

CompareDifferences<-function(SimDiff,RandomSimDiff){

    Wpos<-matrix(FALSE,ncol = ncol(SimDiff),nrow = nrow(SimDiff))
    Wneg<-matrix(FALSE,ncol = ncol(SimDiff),nrow = nrow(SimDiff))

    Wpos[SimDiff>0]<-SimDiff[SimDiff>0]>RandomSimDiff[SimDiff>0]
    Wneg[SimDiff<=0]<-SimDiff[SimDiff<=0]<RandomSimDiff[SimDiff<=0]

    W <- Wpos | Wneg

    return(apply(W,c(1,2),function(x) as.integer(x)))

}

GetNetworkConnectionsFromMask<-function(SimilarityDifferences,mask,TH_MASK=0.95,weighted=TRUE){

    if(weighted){

      SimilarityDifferences<-SimilarityDifferences/max(abs(SimilarityDifferences))

    }else{

      SimilarityDifferences[SimilarityDifferences>=0]<-1

      SimilarityDifferences[SimilarityDifferences<0]<-(-1)


    }

    SimilarityDifferences[mask<TH_MASK]<-0

    NegativeDiffLinks<-SimilarityDifferences[SimilarityDifferences<(-0.000001)]
    PositiveDiffLinks<-SimilarityDifferences[SimilarityDifferences>0.000001]

    print(paste0("Number of Negative Links Detected: ",nnzero(NegativeDiffLinks)))
    print(paste0("Number of Positive Links Detected: ",nnzero(PositiveDiffLinks)))

    return(SimilarityDifferences)
}


GetNumberOfCells <- function(Samples,ClustersAnn,Selection,States,save=TRUE,filename="ncells.rds"){

  samplesComm<-list()

  x<-data.matrix(ClustersAnn)
  n<-row.names(x)
  n<-n[x %in% Selection]

  ncells<-sapply(names(Samples),function(x){

      s<-as.matrix(Samples[[x]])
      sel<-colnames(s) %in% n

      return(sum(sel))

  })

  b<-lapply(States,function(x){

      return(ncells[names(ncells) %in% x])

  })

  if(save){

      saveRDS(b,file=filename)

  }
  return(b)

}

