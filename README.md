# Introduction 
 This is repository created to publish static content to a ststic webiste generated using Hugo framework.
 The website is hosted on Azure platform using Azure blob storage v2 Static webiste feature. It has three main sections
 - Documents
 - Updates
 - Blogs

 [Link to Website](https://r2dldocs.z6.web.core.windows.net/doc-repo/)

# Getting Started
If you are planning to add/modify content on the webiste you need to have the access to the repo and you can either clone it on local system, 
update the content and push back or you can make changes on the go.
For any of the above mentioned method you have to first follow these steps below
1. Clone a branch(if local and first time using it)
2. Create a new branch and follow proper naming conventions[Section-TwoWordAboutAddition/Modification].(for e.g -if you are planning to create new blog on power automate then name the branch as
**Blog-PowerApps**, or if you are adding new document to repo on Branching you can name it as **Docs-Branching**)
3. After you add/modify some content(steps below) please push the code to your branch(if working using git nd not portal)
4. Create a Pull Request(option avialable under VSTS Repos>Pull requests)
5. Please choose *your created branch* into **From** section and *master* branch into **To** section
6. Request all to please add relevant approvers/peers from different stream(By default all squad leads are added as optional reviewers)
7. Once your request is approved and completed, your content will be automatically gets published with the help of azure devops pipeline.(under 5 mins)

## A. Steps to publish new doc under Document section
1. Navigate to the repo **root > content > en > docs > {{Choose from the different streams folder avilable}}**
2. If you are planning to write something under Data-Science Stream, Please choose already existing hierarchy of folder or
   You can create new folder with the _index.md file with below content avialable under quotes at the top
"---
title: "Title of the Folder Structure"
date: 2020-02-28T10:08:56+09:00(Date of the creation)
description: "Some Description about the new hierarchy"
draft: false
collapsible: true
weight: 1 '
---"

3. Now, after this you can create multiple stages under this hierarchy by adding new docs/md files.
   Create a new file under already existing or created folder hierarchy and follow the standard naming convention
   (this will be your content file with md extension). file needs to have  below content avialable under quotes at the top

"---
title: "Title of the document"
description: "Description about the document"
date: 2020-01-28T00:34:51+09:00(Date of creation)
draft: false
weight: -4
---"
"
![RR-Logo](../../../doc-repo/images/Rolls-Royce.jpg)

### Title of the document

Version 1.0

![R2DL-Logo](../../../doc-repo/images/R2DL.jpg)"

"

4. Write the content after the above mentioned quotes and format it properly.
5. If you have any related assets, plase maintain it in a new folder under **static > images > StreamFolderName > TopicFolder**
6. Commit your changes and follow above steps to create PR and your content will be updtaed automatically to portal

## B. Steps to publish new update under Update section
1. Navigate to the repo **root > content > en > updates > {{Choose from the different months files avilable}}**
2. If you are planning to write something for month Jan 2021, Please choose already existing file or
   You can create new one and the name format should be Year_LongMonthName. File needs to have  below content avialable under quotes at the top

"---
title: "Year-Month Name"
description: "Desiptive Updates for Year-Month Name"
date: 2020-12-12T00:10:48+09:00(Creation date)
draft: false
weight: -4
---"


4. Write the content after the above mentioned quotes and format it properly.
5. If you have any related assets, plase maintain it in a new folder under **static > images > Updates > MonthFolderName**
6. Commit your changes and follow above steps to create PR and your content will be updtaed automatically to portal


## C. Steps to publish new blog under Blog section
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


4. Write the content after the above mentioned quotes and format it properly.
5. If you have any related assets, plase maintain it in a new folder under **static > images > Blogs > BlogFolderName**
6. Commit your changes and follow above steps to create PR and your content will be updtaed automatically to portal


# Contribute
I would request each one of you to please start contributing your learning and experience so that it can be shared across.

If you want to learn more about creating good readme files then refer the following
[guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). 
