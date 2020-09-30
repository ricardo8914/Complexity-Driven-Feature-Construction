options(repos=structure(c(CRAN='https://ftp.gwdg.de/pub/misc/cran/')))
options(warn = -1)
suppressMessages(if (!require('bnlearn')) install.packages('bnlearn', dependencies = TRUE)); suppressMessages(library('bnlearn'))
suppressMessages(if (!require('data.table')) install.packages('data.table', dependencies = TRUE)); suppressMessages(library('data.table'))
suppressMessages(if (!require('readr')) install.packages('readr', dependencies = TRUE)); suppressMessages(library('readr'))

#print('Connection with R stablished')
args = commandArgs(trailingOnly=TRUE)

temp_data_path <- args[3]
#print(temp_data_path)
#temp_data_path = '~/Finding-Fair-Representations-Through-Feature-Construction/data/tmp'

#print('Now starting to read dataframe')
dt <- as.data.table(read.csv(paste(paste(temp_data_path,args[4],sep='/'),'.csv',sep = ''), check.names = FALSE))
#print('dataframe was read')

sensitive <- args[1]
target <- args[2]

dt[] <- lapply(dt, function(x) if(is.integer(x) & length(unique(x)) > 10) as.numeric(x) 
               else if(is.integer(x) & length(unique(x)) <= 10)  as.factor(x) 
               else if(is.character(x)) as.factor(x) else x)

black_list = data.frame("from" = sensitive, "to" = target)
new_row = data.frame("from" = target, "to" = sensitive)
black_list = rbind(black_list, new_row)

structure.hc <- hc(dt, blacklist = black_list)

z <- c()
for (i in names(dt)){
  if (i != sensitive & i != target){
    z <- c(z, i)
    
  }
}

if (length(names(dt)) > 2){
  result <- dsep(structure.hc, sensitive, target, z)
} else {
  result <- dsep(structure.hc, sensitive, target)
}


#print('writing markov blanket to file')
write_lines(result, paste(paste(temp_data_path, args[4],sep='/'),".txt", sep = ''))

#print('Done')



