options(warn=-1)
Sys.setenv(LANG = 'en_US.UTF-8')
options(repos=structure(c(CRAN='https://ftp.gwdg.de/pub/misc/cran/')))
suppressMessages(if (!require('bnlearn')) install.packages('bnlearn', dependencies = TRUE)); suppressMessages(library('bnlearn'))
suppressMessages(if (!require('data.table')) install.packages('data.table', dependencies = TRUE)); suppressMessages(library('data.table'))
suppressMessages(if (!require('readr')) install.packages('readr', dependencies = TRUE)); suppressMessages(library('readr'))

#print('Connection with R stablished')
args = commandArgs(trailingOnly=TRUE)

temp_data_path <- args[2]

#print('Now starting to read dataframe')
dt <- as.data.table(read.csv(paste(paste(temp_data_path,args[1],sep='/'),'.csv',sep = ''), check.names = FALSE))
#print('dataframe was read')

dt[] <- lapply(dt, function(x) if(is.integer(x) & length(unique(x)) > 10) as.numeric(x) 
               else if(is.integer(x) & length(unique(x)) <= 10)  as.factor(x) 
               else if(is.character(x)) as.factor(x) else x)

MB <- learn.mb(dt, 'outcome', method = 'gs', alpha = 0.05)

#print('writing markov blanket to file')
write_lines(MB, paste(paste(temp_data_path, args[1],sep='/'),".txt", sep = ''))

#print('Done')



