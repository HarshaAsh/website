# Introduction 
 This is the future version of the [www.harshaash.com](www.harshaash.com) website. www.harshaash.com is my personal blog where I publish blogs on data science. The current active version is published in wordpress. This repository publishes static website using Hugo framework on Netlify. The theme used is [z-doc](https://github.com/zzossig/hugo-theme-zdoc).

 [Link to Website](https://harshaachyuthuni.netlify.app/)

# Getting Started
If you are planning to add/modify content on the webiste you need to have the access to the repo and you can either clone it on local system, 
update the content and push back or you can make changes on the go. To run locally, the command *hugo* will create the html files, and *hugo serve* will host the website locally.  

## Steps to publish new blog under Blog section
1. Navigate to the repo **root > content > en > blog > {{Choose from the different already avilable files}}**
2. If you are planning to write something new, Please create new one and give proper name. 
   File needs to have  below content avialable under quotes at the top

"+++
author = "Name of Author(s)"
title = "Title of the Blog"
date = "2020-12-12" (Creation Date)
description =Description of the blog"
tags = [
    "design-thinking","software-engineering","app-chapter" (Different tags her like this)
]
image = "image path here if you want to add with it"
+++"


4. Write the content after the above mentioned quotes and format it properly. The easiest way is using Jupyter or RShiny to download the file as a markdown file (.md).
5. If you have any related assets, plase maintain it in a new folder under **static > images > Blogs > BlogFolderName**
6. Commit your changes and follow above steps to create PR and your content will be updtaed automatically to portal
