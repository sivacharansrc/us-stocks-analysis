rm(list = ls())
library(dplyr)
library(data.table)
library(foreach)
library(doParallel)
library(bit64)

# Working Dir
path <- "C:/Users/sivac/Documents/Analytics/stock-market-analysis/"
  
files <- list.files(paste0(path, "input/Stocks/"), pattern = "*.txt", full.names = T)[1:300]

# Creating a dummy data frame
df <- data.table(Date = as.character(NA),
                 Open = as.numeric(NA), 
                 High=as.numeric(NA), 
                 Low=as.numeric(NA), 
                 Close=as.numeric(NA), 
                 Volume=as.integer64(NA), 
                 OpenInt=as.integer(NA), 
                 Symbol=as.character(NA))


start_time <- Sys.time()
for (file_name in files) {
  
  if(file.size(file_name) != 0){
    
    # Reading the data
    data <- fread(file_name,
                  stringsAsFactors = F,
                  colClasses = c("character", "numeric", "numeric", "numeric", "numeric",
                                 "integer64", "integer"))
    
    # Creating Symbol column from the file name
    data$Symbol <- gsub(paste0(path, "input/Stocks/"),"",file_name) %>% gsub(".us.txt","",.)
  
    df <- rbindlist(list(df, data))
  }
  df <- df[!is.na(df$Date),]
}

end_time <- Sys.time()
time_diff <- end_time - start_time
time_diff

df$Symbol %>% unique() %>% length()

cluster <- makeCluster(4)
registerDoParallel(cluster)
stopCluster(cluster)

registerDoSEQ()

df <- data.table(Date = as.character(NA),
                 Open = as.numeric(NA), 
                 High=as.numeric(NA), 
                 Low=as.numeric(NA), 
                 Close=as.numeric(NA), 
                 Volume=as.integer64(NA), 
                 OpenInt=as.integer(NA), 
                 Symbol=as.character(NA))

start_time <- Sys.time()
df = foreach(file_name = files, .combine = data.table, .packages = c("data.table","dplyr")) %dopar%
  
  if(file.size(file_name) != 0){
    
    # Reading the data
    data <- fread(file_name,
                  stringsAsFactors = F,
                  colClasses = c("character", "numeric", "numeric", "numeric", "numeric",
                                 "integer64", "integer"))
    
    # Creating Symbol column from the file name
    data$Symbol <- gsub(paste0(path, "input/Stocks/"),"",file_name) %>% gsub(".us.txt","",.)
    
    df <- rbindlist(list(df, data))
  }
  df <- df[!is.na(df$Date),]

end_time <- Sys.time()
time_diff <- end_time - start_time
time_diff

length(unique(df$Symbol))
nrow(df)

file.size("C:/Users/sivac/Documents/Analytics/stock-market-analysis/input/Stocks/a.us.txt")
















# Clensing data frame

df$Date <- as.Date(df$Date)

df[, c("Month", "Year") := list(format.Date(Date, "%B"), format.Date(Date, "%Y"))]
df[, .(Average_Close = mean(Close, na.rm = T)), .(Symbol, Year, Month)]

