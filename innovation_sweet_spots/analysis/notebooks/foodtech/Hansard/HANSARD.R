# install.packages(c(readr, dplyr, XML, progress, stringr, tidyr, tidyverse))


library(readr)
library(dplyr)
library(XML)
library(progress)
library(stringr)
library(tidyr)
library(tidyverse)


# Initial processing ------------------------------------------------------



for (j in c(2000:2022)) {
  
  message("Processing ", j)
  
  atemp <- list.files(path = "debates/", pattern = paste0(j, ".*.xml"))
  dat <- vector("list", length(atemp))
  dat2 <- vector("list", length(atemp))
  dat3 <- vector("list", length(atemp))
  dat4 <- vector("list", length(atemp))
  pb <- progress_bar$new(total = length(atemp),
                         format = ":current/:total :bar :spin")
  
  ## Get speeches
  # re-write this to combine each day when it is done?
  for (i in atemp) {
    pb$tick()
    
    doc <- xmlTreeParse(paste0("debates/",i), useInternalNodes = TRUE)
    
    encode <- getEncoding(doc)
    
    if (is.na(encode)){
      encode <- "UTF-8"
    }
    
    doc <- xmlInternalTreeParse(paste0("debates/",i), useInternalNodes = TRUE, 
                        encoding = encode)
    
    nodes_latest <- getNodeSet(doc, "//publicwhip") 
    latest <- lapply(nodes_latest, function(x) xmlGetAttr(x, "latest", "NA"))
    
    if(latest=="yes"){
      ## Speech content  -----------------------------------------
      nodes <- getNodeSet(doc, "//speech")
      speech <- lapply(nodes, function(x) xmlValue(x, recursive = TRUE))
      speech <- lapply(speech, function(x) iconv(x,  from = "", "UTF-8"))
      speech <- lapply(speech, function(x) str_replace_all(x,  "—", " — "))
      # speech <- lapply(speech, function(x) str_replace(x, "â\u0080\u0094", "—")) 
      speech <- lapply(speech, function(x) str_replace(x, "\n'", " \n"))
      speech <- lapply(speech,
                       function(x) str_replace(x, "([:punct:])'[[:punct:]?]\n", "$1\"\n"))
      
      speech <- lapply(speech, function(x) str_replace_all(x, "\n{2,}", " "))
      speech <- lapply(speech, function(x) str_replace_all(x, "\\n{2,}", " "))
      speech <- lapply(speech, function(x) str_replace_all(x, "\n", "\n "))
      speech <- lapply(speech, function(x) str_replace_all(x, "\\[", " ["))
      speech <- lapply(speech, function(x) str_replace(x, "\\n\\W?$", ""))
      
      id <- lapply(nodes, function(x) xmlAttrs(x)["id"])
      hansard_membership_id <- lapply(nodes, function(x) xmlGetAttr(x, "hansard_membership_id", "NA"))
      speakerid <- lapply(nodes, function(x) xmlGetAttr(x, "speakerid", "NA"))
      person_id <- lapply(nodes, function(x) xmlGetAttr(x, "person_id", "NA"))
      speakername <- lapply(nodes, function(x) xmlGetAttr(x, "speakername", "NA"))
      colnum <- lapply(nodes, function(x) xmlGetAttr(x, "colnum", "NA"))
      time <- lapply(nodes, function(x) xmlGetAttr(x, "time", "NA"))
      url <- lapply(nodes, function(x) xmlGetAttr(x, "url", "NA"))
      speech_class <- lapply(nodes, xpathSApply, ".//div", xmlGetAttr, 'class')
      
      dat[[i]] <- cbind(speech, id, hansard_membership_id, speakerid, person_id,
                        speakername, colnum, time, url, speech_class)
      
      ## Major Headings -----------------------------------------
      nodes2 <- getNodeSet(doc, "//major-heading")
      major_heading <- lapply(nodes2, function(x) xmlValue(x))
      id <- lapply(nodes2, function(x) xmlAttrs(x)["id"])
      nospeaker <- lapply(nodes2, function(x) xmlGetAttr(x, "nospeaker", "NA"))
      colnum <- lapply(nodes2, function(x) xmlGetAttr(x, "colnum", "NA"))
      time <- lapply(nodes2, function(x) xmlGetAttr(x, "time", "NA"))
      url <- lapply(nodes2, function(x) xmlGetAttr(x, "url", "NA"))
      
      
      dat2[[i]] <- cbind(major_heading, id, nospeaker, colnum, time, url)
      
      ## Minor Headings -----------------------------------------
      nodes3 <- getNodeSet(doc, "//minor-heading")
      minor_heading <- lapply(nodes3, function(x) xmlValue(x))
      id <- lapply(nodes3, function(x) xmlAttrs(x)["id"])
      nospeaker <- lapply(nodes3, function(x) xmlGetAttr(x, "nospeaker", "NA"))
      colnum <- lapply(nodes3, function(x) xmlGetAttr(x, "colnum", "NA"))
      time <- lapply(nodes3, function(x) xmlGetAttr(x, "time", "NA"))
      url <- lapply(nodes3, function(x) xmlGetAttr(x, "url", "NA"))
      
      dat3[[i]] <- cbind(minor_heading, id, nospeaker,colnum, time, url)
      
      
      ## Oral Headings -----------------------------------------
      nodes4 <- getNodeSet(doc, "//oral-heading")
      
      if (length(nodes4) > 0) {
        
        oral_heading <- lapply(nodes4, function(x) xmlValue(x))
        id <- lapply(nodes4, function(x) xmlAttrs(x)["id"])
        nospeaker <- lapply(nodes4, function(x) xmlGetAttr(x, "nospeaker", "NA"))
        colnum <- lapply(nodes4, function(x) xmlGetAttr(x, "colnum", "NA"))
        time <- lapply(nodes4, function(x) xmlGetAttr(x, "time", "NA"))
        url <- lapply(nodes4, function(x) xmlGetAttr(x, "url", "NA"))
        
        dat4[[i]] <- cbind(oral_heading, id, nospeaker,colnum, time, url)
        
      }
      
    }
  }
  
  debate <- do.call(rbind.data.frame, dat)
  minor_headings <- do.call(rbind.data.frame, dat3)
  major_headings <- do.call(rbind.data.frame, dat2)
  oral_headings <- do.call(rbind.data.frame, dat4)
  
  
  if (nrow(oral_headings) == 0) {
    oral_headings <- tibble(oral_heading = c(NA))
  }
  
  
  x <- bind_rows(debate, minor_headings, major_headings, oral_headings) %>% 
    as_tibble() %>%
    mutate_all(as.character) %>% 
    mutate(
      sort_id = str_remove_all(
        id, "uk.org.publicwhip/debate/[0-9]{4}-[0-9]{2}-[0-9]{2}[a-z]\\."
      )
    ) %>% 
    separate(sort_id, c("sort1", "sort2")) %>%
    mutate_at(vars(matches("sort")), as.numeric) %>% 
    mutate_all(na_if, "") %>%
    mutate_all(na_if, "NA") %>%
    mutate_all(na_if, "NULL") %>%
    mutate(date = gsub("uk.org.publicwhip/debate/", "", id),
           date = str_sub(date, 1,10),
           date = as.Date(date),
           year = lubridate::year(date)) %>% 
    arrange(date, sort1, sort2) %>%
    group_by(date) %>%
    fill(time, major_heading) %>% 
    group_by(date, major_heading) %>%
    fill(minor_heading) %>% 
    group_by(date, major_heading, minor_heading) %>%
    fill(oral_heading) %>% 
    ungroup() %>% 
    filter(!is.na(speech), date >= as.Date("1979-05-07"))  %>% 
    mutate(
      speech_class = case_when(
       str_detect(speech, 
                  "^The House divided: Ayes [0-9]{1,3}, Noes [0-9]{1,3}\\.") ~ "Division",
       str_detect(speech,  "^Question,? ") ~ "Procedural",
        str_detect(speech, "^Mr. Speaker forthwith")~ "Procedural",
       str_detect(speech, "^Resolved, That this House")~ "Procedural",
       str_detect(speech, "^Resolved,That this House,")~ "Procedural",
       major_heading == "MEMBERS SWORN"~ "Procedural",
       major_heading == "Preamble" ~ "Procedural",
      TRUE ~ speech_class
        )) %>% 
    mutate(major_heading = iconv(major_heading,  from = "", "UTF-8"),
           minor_heading = iconv(minor_heading,  from = "", "UTF-8"),
           oral_heading = iconv(oral_heading,  from = "", "UTF-8")) %>% 
    group_by(date, speech_class, major_heading, minor_heading, oral_heading)
  
  save_name <- paste0("debate-single-years/", j, ".rds")
  
  write_rds(x, file = save_name)
  
}

FINAL <- list.files("debate-single-years", pattern = "*.rds", full.names = TRUE) %>%
map(readRDS) %>%
bind_rows()
write_csv(FINAL, 'HANSARD.csv')
