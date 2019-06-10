# install dependency
install.packages("BiocManager")
BiocManager::install('GenomicRanges')
BiocManager::install('rtracklayer')
BiocManager::install('BSgenome')
BiocManager::install('BSgenome.Hsapiens.UCSC.hg19.masked')
install.packages('ROCR')
install.packages('kernlab')
install.packages('seqinr')

# install gkm-SVM
system("git clone https://github.com/mghandi/gkmSVM.git")
system("R CMD INSTALL gkmSVM")


BiocManager::install("Rsamtools")
