options(repos=structure(c(CRAN='https://ftp.gwdg.de/pub/misc/cran/')))
#suppressMessages(if (!require('devtools')) install.packages('devtools', dependencies = TRUE)); suppressMessages(library('devtools'))
#install_github("ericstrobl/RCIT")
#library(RCIT)
options(warn = -1)
#suppressMessages(if (!require('MASS')) install.packages('MASS', dependencies = TRUE)); suppressMessages(library('MASS'))
#suppressMessages(if (!require('momentchi2')) install.packages('momentchi2', dependencies = TRUE)); suppressMessages(library('momentchi2'))
suppressMessages(if (!require('bnlearn')) install.packages('bnlearn', dependencies = TRUE)); suppressMessages(library('bnlearn'))
suppressMessages(if (!require('data.table')) install.packages('data.table', dependencies = TRUE)); suppressMessages(library('data.table'))
suppressMessages(if (!require('readr')) install.packages('readr', dependencies = TRUE)); suppressMessages(library('readr'))
suppressMessages(if (!require('stringr')) install.packages('stringr', dependencies = TRUE)); suppressMessages(library('stringr'))
suppressMessages(if (!require('comprehenr')) install.packages('comprehenr', dependencies = TRUE)); suppressMessages(library('comprehenr'))
#suppressMessages(if (!require('CondIndTests')) install.packages('CondIndTests', dependencies = TRUE)); suppressMessages(library('CondIndTests'))
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
#library('Rgraphviz')

args = commandArgs(trailingOnly=TRUE)

temp_data_path <- args[1]

#temp_data_path <- '~/Finding-Fair-Representations-Through-Feature-Construction/data/Intermediate_results'

dt <- as.data.table(read.csv(paste(paste(temp_data_path,args[2],sep='/'),'.csv',sep = ''), check.names = FALSE))
#dt <- as.data.table(read.csv(paste(paste(temp_data_path,'complete_extended_df',sep='/'),'.csv',sep = ''), check.names = FALSE))

dt[] <- lapply(dt, function(x) if(is.integer(x) & length(unique(x)) > 10) as.numeric(x) 
               else if(is.integer(x) & length(unique(x)) <= 10)  as.factor(x) 
               else if(is.character(x)) as.factor(x) else x)

#dt[ ,`:=`(nativecountry = NULL, relationship = NULL, race = NULL)]

s <- c()
a <- c()
x <- c()
for (i in names(dt)){
  if (substr(i, 1, 2) == 's_'){
    s <- c(s, i)
  } else if (substr(i, 1, 2) == 'a_'){
    a <- c(a, i)
  }
  else if (substr(i, 1, 2) == 'x_'){
    x <- c(x, i)
  }
  else{
    target <- i
  }
}

#X_features <- dt[, ..x]
#S_features <- dt[, ..s]
#A_features <- dt[, ..a]
#cit <- RCIT(as.matrix(X_features), as.matrix(S_features), as.matrix(A_features), num_f = 500)

black_list = setNames(data.frame(matrix(ncol = 2, nrow = 0)), c("from", "to"))
for (i in names(dt)){
  if (i != target){
    new_row = data.frame("from" = target, "to" = i)
    black_list = rbind(black_list, new_row)
  }
  #if (i %in% s){
  #  new_row = data.frame("from" = i, "to" = target)
  #  black_list = rbind(black_list, new_row)
  #}
}

#a <- to_vec(for(x in names(dt)) if(!x %in% s) x)


#start_time_hc <- Sys.time()
#structure.hc <- tabu(dt, debug=TRUE, blacklist = black_list, maxp = 5, tabu=20)
structure.hc <- hc(dt, blacklist = black_list, maxp=2)
#end_time_hc <- Sys.time() - start_time_hc

#structure.fast_iamb <- fast.iamb(dt, blacklist = black_list)

#gR = graphviz.plot(structure.hc, shape = 'ellipse', layout = "fdp", groups = list(s, a, x), 
#              highlight = list(arcs = rbind(outgoing.arcs(structure.hc, 's_sex'), 
#                                            outgoing.arcs(structure.hc, 's_marital_status')), 
#                               nodes = s, col = "tomato", fill = "tomato",lwd = 1, lty = "dashed"), render = FALSE)

#node.attrs = nodeRenderInfo(gR)
#node.attrs$col[target] = "darkblue"
#node.attrs$fill[target] = "lightblue"
#nodeRenderInfo(gR) = node.attrs

#graphviz.plot(structure.fast_iamb, layout = "dot")

c1 <- c()
for (i in x){
  r <- c()
  for (d in s){
    if (dsep(structure.hc, i, d, a)){
      r <- c(r, TRUE)
    } else{
      r <- c(r, FALSE)
    }
  }
  if (all(r)){
    c1 <- c(c1, i)
  }
}

x_trunc <- to_vec(for(i in x) if(!i %in% c1) i)

a_c1 <- c(a, c1)
c2 <- c()
for (i in x_trunc){
  if (dsep(structure.hc, i, target, a_c1)){
    c2 <- c(c2, i)
  }
}

x_selected <- c(c1, c2)
#for (i in x_selected){
#  node.attrs = nodeRenderInfo(gR)
#  node.attrs$col[i] = "darkblue"
#  node.attrs$fill[i] = "green"
#  nodeRenderInfo(gR) = node.attrs
#}

#renderGraph(gR)


write_lines(x_selected, paste(paste(temp_data_path, args[2],sep='/'),".txt", sep = ''))



