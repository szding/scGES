options(warn=-1) 
library(dplyr)
library(cowplot)
library(patchwork)
library(reticulate)
.libPaths(c("/home/dszspur/.conda/envs/py38/lib/R",.libPaths())) 
Sys.setenv(RETICULATE_PYTHON_ENV="/home/dszspur/.conda/envs/py38")
Sys.setenv(RETICULATE_PYTHON="/home/dszspur/.conda/envs/py38/bin/python")
source("/mnt/mechanical_drive/DATA/dszspur/scGES/scGES/R_untils.R")
df_pseudotime_list=list()
old=theme_set(theme_bw()+theme(strip.background = element_rect(fill="white"),
                               panel.background = element_blank(),
                               legend.background = element_blank(),
                               panel.grid =element_blank()))
maprules=c("ss2"="ss2", "mars"="mars")

dir  =  "/mnt/mechanical_drive/DATA/dszspur/scGES/"

ad=import("anndata",convert = FALSE)
adata=ad$read_h5ad(paste0(dir, "data/monocle_scGES.h5ad"))
adata

cell.meta.data=py_to_r(adata$obs)
cell.meta.data$dataset_batch=plyr::mapvalues(cell.meta.data$batch,names(maprules),maprules)
gene_ann0=py_to_r(adata$var)
gene_ann=data.frame(gene_short_name = make.unique(rownames(gene_ann0)),
                    VarianceType=gene_ann0$`Variance Type`,
                    row.names = make.unique(rownames(gene_ann0)))

mtx=t(py_to_r(adata$layers['atlas ALL denoised']))
colnames(mtx)=rownames(cell.meta.data)
rownames(mtx)=rownames(gene_ann)
mtx_sizefactor=1e4/colSums(mtx)
mtx[1:5,1:5]

genes = rownames(gene_ann)
hvg_genes <- gene_ann %>% 
  filter(VarianceType == "HVG") %>% 
  pull(gene_short_name)
lvg_genes <- gene_ann %>% 
  filter(VarianceType == "LVG") %>% 
  pull(gene_short_name)

# Monocle3
cds <- new_cell_data_set(mtx, cell_metadata = cell.meta.data,gene_metadata =gene_ann)
cds <- preprocess_cds(cds, num_dim = 20, method="PCA", norm_method="log", verbose = F)
cds <- reduce_dimension(cds,reduction_method = "UMAP",preprocess_method="PCA",verbose = F)
cds <- cluster_cells(cds,reduction_method ="UMAP",cluster_method = "leiden",verbose = F, k = 16)
cds <- learn_graph(cds, use_partition = T,verbose = F)

colData(cds)$clusters=cds@clusters$UMAP$clusters
p1=plot_cells(cds,color_cells_by = "partition",label_cell_groups = F)+theme(legend.position = "top")
p2=plot_cells(cds,color_cells_by = "clusters",label_cell_groups=F,graph_label_size=2, label_leaves=F,label_branch_points=F)+theme(legend.position = "top")
p=cowplot::plot_grid(p1,p2,align = "h",ncol = 3)
options(repr.plot.width=10, repr.plot.height=4)
p


ids=get_earliest_principal_node(cds, cluster=c("3"))
cds <- order_cells(cds, root_pr_nodes=ids)

options(repr.plot.width=6, repr.plot.height=4)
plot_cells(cds,color_cells_by = "pseudotime")


cds_sub.1 <- choose_graph_segments(cds)
cds_sub.2 <- choose_graph_segments(cds)


library(egg)
colData(cds)$pseudotime=pseudotime(cds)
colData(cds)$Pseudotime=colData(cds)$pseudotime/max(colData(cds)$pseudotime,na.rm = T)
df_den=pData(cds)[,c("Pseudotime", "dataset_batch", "cell_type")]
df_den=as.data.frame(df_den[!is.infinite(df_den$Pseudotime),])
set.seed(10)

theme_use=theme(legend.text = element_text(size=16),
                legend.title = element_text(size=20))
ll = 18
p_scGES_denoised_all_0=plot_cells(cds,color_cells_by = "clusters",label_cell_groups=F, graph_label_size=4, label_leaves=F,label_branch_points=F)+
  theme(legend.position = "top",
        legend.text = element_text(size = ll),
        legend.title = element_text(size = ll),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14))

p_scGES_denoised_all_1=plot_cells(cds,color_cells_by = "dataset_batch",,graph_label_size=0,alpha=1,cell_size = 0.6)+
  guides(colour = guide_legend(override.aes = list(alpha=0.7, size=5)))+theme_use+
  theme(legend.position = "top",
        legend.text = element_text(size = ll),
        legend.title = element_text(size = ll),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14))

p_scGES_denoised_all_2=plot_cells(cds,color_cells_by = "Pseudotime",label_branch_points=T,graph_label_size=2, alpha=1,cell_size = 0.6)+
  theme(legend.position = "top",
        legend.title = element_text(vjust = 0.2),
        legend.text = element_text(angle=-50 , size = 9),
        legend.key.height = unit(0.5,"cm"),
        legend.key.width = unit(1,"cm"),
        legend.spacing.y = unit(0.5, "cm"),
        legend.margin = margin(t = 10, r = 0, b = 0, l = 0))+
  guides(color = guide_colourbar(label.position = "top"))+
  theme(legend.position = "top",
        legend.text = element_text(size = ll),
        legend.title = element_text(size = ll),
        axis.text.x = element_text(size = 12), 
        axis.text.y = element_text(size = 12), 
        axis.title.x = element_text(size = 14), 
        axis.title.y = element_text(size = 14))  

# Choose branch
df_den_MEP <- df_den[colnames(cds_sub.1),]
df_den_GMP <- df_den[colnames(cds_sub.2), ]

p_scGES_denoised_all_31=ggplot(data=df_den_GMP) + geom_density(aes(x=Pseudotime,fill=dataset_batch), alpha=0.6)+
  scale_y_continuous(expand = c(0,0))+
  scale_x_continuous(expand = c(0,0))+
  theme(legend.position = "top",
        plot.margin = unit(c(0, 1, 0, 0), "cm"),
        legend.text = element_text(size = ll),
        legend.title = element_text(size = ll),
        axis.text.x = element_text(size = 12), 
        axis.text.y = element_text(size = 12), 
        axis.title.x = element_text(size = 14), 
        axis.title.y = element_text(size = 14))  

p_scGES_denoised_all_32=ggplot(data=df_den_MEP) + geom_density(aes(x=Pseudotime,fill=dataset_batch), alpha=0.6)+
  scale_y_continuous(expand = c(0,0))+
  scale_x_continuous(expand = c(0,0))+
  theme(legend.position = "top",
        plot.margin = unit(c(0, 1, 0, 0), "cm"),
        legend.text = element_text(size = ll),
        legend.title = element_text(size = ll),
        axis.text.x = element_text(size = 12), 
        axis.text.y = element_text(size = 12),  
        axis.title.x = element_text(size = 14), 
        axis.title.y = element_text(size = 14))  

p_monocle_scGES_denoised_all=egg::ggarrange(p_scGES_denoised_all_0, p_scGES_denoised_all_1, p_scGES_denoised_all_2,  ncol=3, draw=F)

options(repr.plot.width=23, repr.plot.height=4)
p_monocle_scGES_denoised_all <- ggdraw(p_monocle_scGES_denoised_all)
p_monocle_scGES_denoised_all

ggsave(filename = paste0(dir, "plot/scGES_integration_pseudo-new.png"), plot = p_monocle_scGES_denoised_all, width = 15, height = 4, dpi = 300)


###########################################################################################
#########################  HVG &LVG
###########################################################################################
cds_exprs=as.matrix(SingleCellExperiment::counts(cds)[genes,]) 
df0=data.frame(cbind(pseudotime=pData(cds)$Pseudotime, log1p(t(cds_exprs)*mtx_sizefactor)))
df0$UMAP_1=reducedDims(cds)$UMAP[,1]
df0$UMAP_2=reducedDims(cds)$UMAP[,2]
df0$BatchID=pData(cds)$dataset_batch
df0=df0[is.finite(df0$pseudotime),]
df0=df0[order(df0$pseudotime,decreasing = F),,drop=F]
df0$x=df0$pseudotime/max(df0$pseudotime)
df_pseudotime_list$scGES_denoised_all=df0

df_MEP = df0[rownames(df_den_MEP), ]
df_GMP = df0[rownames(df_den_GMP), ]

p_umap <- get_plot4_HSC_umap(df00 = df_MEP, genes = c("APOE","KLF1","CEBPA"))   #
p_umap_1 <- ggdraw(p_umap) 
p_pseudo <- get_plot4_HSC_pseudo(df00 = df_MEP, genes = c("APOE","KLF1","CEBPA"))
p_pseudo_1 <- ggdraw(p_pseudo) 

p_umap <- get_plot4_HSC_umap(df00 = df_GMP, genes = c("APOE","KLF1","CEBPA"))   #
p_umap_2 <- ggdraw(p_umap) 

p_pseudo <- get_plot4_HSC_pseudo(df00 = df_GMP, genes = c("APOE","KLF1","CEBPA"))
p_pseudo_2 <- ggdraw(p_pseudo) 

combined_plot <- plot_grid(p_umap_1, p_umap_2, nrow = 1)
options(repr.plot.width=10, repr.plot.height=8)
combined_plot
output_file <- paste0(dir, "plot/scGES_HVGgene-UMAP.png")
ggsave(filename = output_file, plot = combined_plot, width = 15, height = 3, dpi = 600)

combined_plot <- plot_grid(p_pseudo_1, p_pseudo_2, nrow = 1)
options(repr.plot.width=10, repr.plot.height=8)
combined_plot
output_file <- paste0(dir, "plot/scGES_HVGgene-pseudo.png")
ggsave(filename = output_file, plot = combined_plot, width = 15, height = 3, dpi = 600)


p_umap <- get_plot4_HSC_umap(df00 = df_MEP, genes = c("LIMS1",  "CAR2", "TRF"))  #  
p_umap_1 <- ggdraw(p_umap) 
p_pseudo <- get_plot4_HSC_pseudo(df00 = df_MEP, genes = c("LIMS1",  "CAR2", "TRF"))
p_pseudo_1 <- ggdraw(p_pseudo)

p_umap <- get_plot4_HSC_umap(df00 = df_GMP, genes = c("LIMS1",  "CAR2", "TRF"))  #  
p_umap_2 <- ggdraw(p_umap) 
p_pseudo <- get_plot4_HSC_pseudo(df00 = df_GMP, genes = c("LIMS1",  "CAR2", "TRF"))
p_pseudo_2 <- ggdraw(p_pseudo)

combined_plot <- plot_grid(p_umap_1, p_umap_2, nrow = 1)
options(repr.plot.width=10, repr.plot.height=8)
combined_plot
output_file <-paste0(dir, "plot/scGES_LVGgene-UMAP.png")
ggsave(filename = output_file, plot = combined_plot, width = 15, height = 3, dpi = 600)

combined_plot <- plot_grid(p_pseudo_1, p_pseudo_2, nrow = 1)
options(repr.plot.width=10, repr.plot.height=8)
combined_plot
output_file <- paste0(dir, "plot/scGES_LVGgene-pseudo.png")
ggsave(filename = output_file, plot = combined_plot, width = 15, height = 3, dpi = 600)

