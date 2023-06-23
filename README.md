# Assessment of pathogenic potential for *Vibrio parahaemolyticus* achieved by high-performance machine learning models
*Vibrio parahaemolyticus* (*Vp*) is a Gram-negative, rod-shaped and halophilic pathogenic bacterium. The wide presence of *Vp* in different production stages of seafood has generated negative impacts on both public health and the seafood industry worldwide, which results in 45,000 illnesses annually (CDC, 2021). The Economic Research Service of the United States Department of Agriculture (USDA ERS) estimated that the total cost of illnesses caused by *Vp* was $40,682,312 in 2013 (USDA, 2013). Such undesirable real-world evidence underscores the imperative needs to control *Vp* in different chained stages of food systems (Liu et al., 2023). Especially, *Vp* is one of the biggest concerns in shellfish industry of California as evidenced by previous *Vp* prevalence studies (DePaola et al., 1990; Dickinson et al., 2013). 

Virulence characterization is the frontline stage in pathogen control, which has undergone a revolution with the advent and increasing accessibility of next-generation sequencing (NGS) (Lecuit & Eloit, 2014). The pathogenic potential of *Vp* is multifactorial, encompassing various virulence factors including but not limited to thermostable direct hemolysin (TDH), TDH-related hemolysin (TRH), and two type III secretion systems (T3SS1 and T3SS2). However, the varity of virulence factors and their intricate interplays resulting in actual virulence of *Vp* remains largely elusive at this moment. It is evidenced that environmental *Vp* isolates also harbored tdh gene, which is previously considered as the key criteria in pathogenicity determination and evaluation of clinical isolates (Paranjpye et al., 2012). Besides, there exist several limitations on current methods to characterize virulence of *Vp*. 1) the standardized protocols for virulence characterization of Vp is usually time consuming, labor intensive and requires well-trained bioinformatic expertise. A more time-effective virulence characterization method is imperatively needed. 2) Since *Vp* is a non-model microorganism, a large number of genes in *Vp* pangenome still lack of accurate and informative annotations based on Kyoto Encyclopedia of Genes and Genomes and Gene Ontology databases and the correlation between these unidentified or putative functional genes and actual pathogenic virulence has not been investigated at this moment. These aforementioned knowledge gaps hinder the virulence characterization of *Vp*, which serves as the first stage in the development of effective interventions to prevent and control *Vp* infections. 

The utilization of machine learning (ML) methods that leverage whole-genome sequencing (WGS) data has garnered considerable attention and interest in recent years (Libbrecht & Noble, 2015). This heightened interest can be attributed primarily to the remarkable advancements made in both ML algorithms and sequencing technologies, which have revolutionized the field of genomics. ML, characterized by their ability to automatically learn patterns and make predictions from data, have emerged as powerful tools to analyze and interpret WGS data (Mahesh, 2020). ML algorithms can identify complex relationships and patterns within the genome, enabling the discovery of genetic variations, functional elements, and associations with virulence serotypes. In recent decade, application of deep learning models (neural network/convolution neural network) has been exponentially applied into different fields because of its remarkable performance in various tasks such as image recognition, natural language processing, and general data analysis (Greener et al., 2022). It is promising to bring these powerful deep learning tools onto the field of microbial food safety. The integration of level-1 ML and level-2 deep learning methods with WGS data has the potential to enhance our understanding of the pathogenicity basis of bacterial pathogens, facilitate later stage decision making process, and contribute to advancements in final pathogen control.

**Therefore, the goal of this study is to address the urgent need for effective control measures against *Vp* by fast and accurate fundamental virulence prediction using level-1 ML and level-2 deep learning algorithms. This study aims to leverage ML methods and WGS data to advance the characterization *Vp* pathogenicity and benefit the modern public health.

Objectives
1.	Establishing and evaluating level-1 machine learning models (Random Forest, K-Nearest Neighbor, K-means, Support Vector Machine, and Principal Component Analysis)
2.	Establishing and evaluating level-2 deep learning models (Neural Network, Convolutional Neural Network)
3.	Validating established machine learning models using newly generated whole genome sequencing data of Vp isolated from oyster hatchery seawater samples

**Reference**

Bankevich, A., Nurk, S., Antipov, D., Gurevich, A. A., Dvorkin, M., Kulikov, A. S., Lesin, V. M., Nikolenko, S. I., Pham, S., & Prjibelski, A. D. (2012). SPAdes: a new genome assembly algorithm and its applications to single-cell sequencing. Journal of computational biology, 19(5), 455-477. 

CDC. (2021). Vibrio species causing Vibriosis. Questions and answers Available at https://www.cdc.gov/vibrio/index.html Accessed on April 12, 2022. Retrieved July, 20th, 2021 

DePaola, A., Hopkins, L., Peeler, J., Wentz, B., & McPhearson, R. (1990). Incidence of Vibrio parahaemolyticus in US coastal waters and oysters. Applied and environmental microbiology, 56(8), 2299-2302. 

Dickinson, G., Lim, K.-y., & Jiang, S. C. (2013). Quantitative microbial risk assessment of pathogenic vibrios in marine recreational waters of Southern California. Applied and environmental microbiology, 79(1), 294-302. 

Greener, J. G., Kandathil, S. M., Moffat, L., & Jones, D. T. (2022). A guide to machine learning for biologists. Nature Reviews Molecular Cell Biology, 23(1), 40-55. 

Gurevich, A., Saveliev, V., Vyahhi, N., & Tesler, G. (2013). QUAST: quality assessment tool for genome assemblies. Bioinformatics, 29(8), 1072-1075. 

Im, H., Hwang, S.-H., Kim, B. S., & Choi, S. H. (2021). Pathogenic potential assessment of the Shiga toxin–producing Escherichia coli by a source attribution–considered machine learning model. Proceedings of the National Academy of Sciences, 118(20), e2018877118. 

Kaysner, C. A., DePaola, A., & Jones, J. (2004). BAM chapter 9: Vibrio. Bacteriological analytical manual (BAM). 

Lecuit, M., & Eloit, M. (2014). The diagnosis of infectious diseases by whole genome next generation sequencing: a new era is opening. Frontiers in cellular and infection microbiology, 4, 25. 

Libbrecht, M. W., & Noble, W. S. (2015). Machine learning applications in genetics and genomics. Nature Reviews Genetics, 16(6), 321-332. 

Liu, Z., Zhou, Y., Wang, H., Liu, C., & Wang, L. (2023). Recent advances in understanding the survival mechanisms of Vibrio parahaemolyticus (under review)
. Applied Environmental and Microbiology. 

Mahesh, B. (2020). Machine learning algorithms-a review. International Journal of Science and Research (IJSR).[Internet], 9, 381-386. 

Paranjpye, R., Hamel, O. S., Stojanovski, A., & Liermann, M. (2012). Genetic diversity of clinical and environmental Vibrio parahaemolyticus strains from the Pacific Northwest. Applied and environmental microbiology, 78(24), 8631-8638. 

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., & Antiga, L. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32. 

Seemann, T. (2014). Prokka: rapid prokaryotic genome annotation. Bioinformatics, 30(14), 2068-2069. 
USDA. (2013). Cost of foodborne illness estimates for Vibrio parahaemolyticus Available at https://www.ers.usda.gov/webdocs/DataFiles/48464/Vibrio_para.xlsx?v=0 Accessed on August 10th, 2021. 

Varoquaux, G., Buitinck, L., Louppe, G., Grisel, O., Pedregosa, F., & Mueller, A. (2015). Scikit-learn: Machine learning without learning the machinery. GetMobile: Mobile Computing and Communications, 19(1), 29-33. 

Wood, D. E., Lu, J., & Langmead, B. (2019). Improved metagenomic analysis with Kraken 2. Genome biology, 20, 1-13. 


