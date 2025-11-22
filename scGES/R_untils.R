suppressPackageStartupMessages(library(reticulate))
suppressPackageStartupMessages(library(monocle3))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggjoy))
suppressPackageStartupMessages(library(VGAM))
suppressPackageStartupMessages(library(knitr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(kableExtra))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(SeuratObject))
suppressPackageStartupMessages(library(Seurat))

datadirpath="./"
knitr::opts_chunk$set(echo=T)


get_earliest_principal_node <- function(cds, cluster=c("1","5")){
  root_pr_nodes=sapply(cluster,function(ii){
    cell_ids <- which(colData(cds)[, "clusters"] %in%ii)
    
    closest_vertex <-cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex
    
    closest_vertex <- as.matrix(closest_vertex[colnames(cds), ])
    root_pr_nodes <-igraph::V(principal_graph(cds)[["UMAP"]])$name[as.numeric(names(which.max(table(closest_vertex[cell_ids,]))))]
  })
  root_pr_nodes
}

get_draw_plot=function(plot_id=1, plist0){
  x=0.02
  y=1-plot_id/5
  width=0.98
  height=1/5-0.01 # total number of figures is 12
  pp=draw_plot(egg::ggarrange(plots=plist0, nrow = 1,draw = F,newpage = F), x = x, y = y, width = width, height = height)
  #draw_label(labels[plot_id],x=x,y=y+1/6,hjust =1,vjust = 0.5,size = 35)
  return(pp)
}

get_label_pos=function(plot_id=1){
  x=0
  y=1-plot_id/5
  #draw_label(labels[plot_id],x=x,y=y+1/6,hjust =1,vjust = 0.5,size = 35)
  return(c(x,y+1/5-1/30))
}

get_title_pos=function(plot_id=1){
  x=0.015
  y=1-plot_id/5
  #draw_label(labels[plot_id],x=x,y=y+1/6,hjust =1,vjust = 0.5,size = 35)
  return(c(x,y+1/10-1/60))
}

get_plot_list=function(x,y){
  x0=rep(list(),length=length(x)+length(y))
  x0[1:length(x)]=x[1:3]
  x0[(length(x)+1):length(x0)]=y[1:2]
  return(x0)
}

get_p_new =function(x,x2=NULL) {
  res1=sapply(x,function(i0) ifelse(i0<=0,"<2.2e-16",scales::scientific(i0,digits = 3))) 
  res2=rep(c(""),length=length(res1))
  if(!is.null(x2)){
    res2=round(x2,3) 
  }
  return(paste0(res2," (",res1,")"))
}

BatchKL=function(df,dimensionData=NULL,replicates=200,n_neighbors=100,n_cells=100,batch="BatchID"){

  if (is.null(dimensionData)){
    tsnedata=as.matrix(df[,c("tSNE_1","tSNE_2")])
  }else{
    tsnedata=as.matrix(dimensionData)
  }
  batchdata=factor(as.vector(df[,batch]))
  table.batchdata=as.matrix(table(batchdata))[,1]
  tmp00=table.batchdata/sum(table.batchdata)#proportation of population
  n=dim(df)[1]
  KL=sapply(1:replicates,function(x){
    bootsamples=sample(1:n,n_cells)
    nearest=nabor::knn(tsnedata,tsnedata[bootsamples,],k=min(5*length(tmp00),n_neighbors))
    KL_x=sapply(1:length(bootsamples),function(y){
      id=nearest$nn.idx[y,]
      tmp=as.matrix(table(batchdata[id]))[,1]
      tmp=tmp/sum(tmp)
      return(sum(tmp*log2(tmp/tmp00),na.rm = T))
    })
    return(mean(KL_x,na.rm = T))
  })
  return(KL)
}

Convert_to_seurat3=function(adata){

  suppressPackageStartupMessages(library("Seurat"))
  mtx=py_to_r(adata$X$T) #adata$X$T$tocsc()
  cellinfo=py_to_r(adata$obs)
  geneinfo=py_to_r(adata$var)
 
  colnames(mtx)=rownames(cellinfo)
  rownames(mtx)=rownames(geneinfo)
  obj=CreateSeuratObject(mtx, meta.data = cellinfo[,!colnames(cellinfo)%in%c("n_genes","n_counts"),drop=F],min.features  = 1)
  return(obj)
}
getwd()


get_plot4_HSC <- function(df00, genes = c("CAR1", "FLT3", "MPO")) {
  # Create UMAP plots
  umap_plots <- lapply(genes, function(gene) {
    ggplot() + 
      geom_point(data = df00, aes(x = UMAP_1, y = UMAP_2, color = .data[[gene]]), size = 0.01) +
      scale_color_gradient(low = "grey", high = "red") +
      theme(legend.position = "top") +
      guides(color = guide_colorbar(title.vjust = 0.7))
  })
  
  # Create pseudotime plots
  pseudotime_plots <- lapply(genes, function(gene) {
    ggplot(data = df00, aes(x = pseudotime, y = .data[[gene]])) +
      geom_point(aes(color = BatchID), size = 0.01) +
      guides(color = guide_legend(override.aes = list(size = 5))) +
      geom_smooth(aes(color = BatchID), method = "gam", formula = y ~ s(x, bs = "cs")) +
      geom_smooth(color = "black", method = "gam", formula = y ~ s(x, bs = "cs"), size = 0.5) +
      ggtitle("") +
      xlab("Pseudotime") +
      theme(legend.position = "top",
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            legend.text = element_text(size = 15, face = "bold"),
            plot.margin = unit(c(0, 0.5, 0, 0.5), "cm"),
            legend.title = element_blank()) +
      scale_color_brewer(palette = "Set2")
  })
  
  # Combine all plots
  all_plots <- c(umap_plots, pseudotime_plots)
  p <- egg::ggarrange(plots = all_plots, ncol = length(genes) * 2, draw = FALSE)
  return(p)
}

get_plot4_HSC_umap <- function(df00, genes = c("CAR1", "FLT3", "MPO")) {
  # Create UMAP plots
  umap_plots <- lapply(genes, function(gene) {
    ggplot() + 
      geom_point(data = df00, aes(x = UMAP_1, y = UMAP_2, color = .data[[gene]]), size = 0.01) +
      scale_color_gradient(low = "grey", high = "red") +
      theme(legend.position = "top") + 
      guides(color = guide_colorbar(title.vjust = 0.7))
  })
  
  all_plots <- c(umap_plots)
  p <- egg::ggarrange(plots = all_plots, ncol = length(genes), draw = FALSE)
  return(p)
}
              
get_plot4_HSC_pseudo <- function(df00, genes = c("CAR1", "FLT3", "MPO")) {
  # Create pseudotime plots
  pseudotime_plots <- lapply(genes, function(gene) {
    ggplot(data = df00, aes(x = pseudotime, y = .data[[gene]])) +
      geom_point(aes(color = BatchID), size = 0.01) +
      guides(color = guide_legend(override.aes = list(size = 5))) +
      geom_smooth(aes(color = BatchID), method = "gam", formula = y ~ s(x, bs = "cs")) +
      geom_smooth(color = "black", method = "gam", formula = y ~ s(x, bs = "cs"), size = 0.5) +
      ggtitle("") +
      xlab("Pseudotime") +
      theme(legend.position = "top",
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            legend.text = element_text(size = 15, face = "bold"),
            plot.margin = unit(c(0, 0.5, 0, 0.5), "cm"),
            legend.title = element_blank()) +
      scale_color_brewer(palette = "Set2")
  })
  
  # Combine all plots
  all_plots <- c(pseudotime_plots)
  p <- egg::ggarrange(plots = all_plots, ncol = length(genes) , draw = FALSE)
  return(p)
}