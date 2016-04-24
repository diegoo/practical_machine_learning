library(knitr)
library(markdown)
library(rmarkdown)
knit('project.Rmd', 'project.md'); markdownToHTML('project.md', 'project.html')
