# Guide for downloading hansard data

1. Create a working directory.
3. Navigate to this working directory in the terminal
4. Run Rsync code in the terminal to download speeches `rsync -az --exclude='*s19*'  --progress --exclude '.svn' --exclude 'tmp/' --relative  data.theyworkforyou.com::parldata/scrapedxml/debates/  .`

This will download all debates from 2000+ (technically all debates that don't start with 19..

4. Navigate to the sub-directory scrapedxml
3. Create a folder in scraped xml called `debate-single-years`
4. Open R
5. Run command `setwd('WORKING_DIRECTORY/scrapedxml/')
6. Run HANSARD.R, making sure to comment/uncomment the install.packages
7. A file called 'HANSARD.csv' is created, this contains all speeches with additional information
