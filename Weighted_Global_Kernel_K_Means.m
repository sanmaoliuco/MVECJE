function [Cluster_elem,Clustering_error]=Weighted_Global_Kernel_K_Means(K,Dataset_Weights,Total_clusters,Display)
%Dataset size.
Data_num=size(K,1);

%Store the optimal clustering error when searching for 1,2,...,Total_clusters.
Best_error=inf(1,Total_clusters);

%Store the assignment of points to clusters corresponding to the optimal solution for 1,2,...,Total_clusters.
Best_clusters=ones(Data_num,Total_clusters);

%Find one cluster solution.
[Best_clusters(:,1),Best_error(1)]=Weighted_Kernel_K_Means(Best_clusters(:,1),K,Dataset_Weights,1,Display);

%Find 2,...,Total_Clusters solutions.
for m=2:Total_clusters
   
    %Solve the m clustering problem by trying all datapoints as possible initializations for the m-th cluster.
    for n=1:Data_num
        
        %Place the m-th cluster initially at point n.
        %The other clusters are initiallized using the solution to the m-1 clustering problem.
        Cluster_elem=Best_clusters(:,m-1);
        Cluster_elem(n)=m;
        
        if strcmp(Display,'details') || strcmp(Display,'nutshell')
            fprintf('\n\nSearching for %d clusters.Intitially placing %dth cluster at point %d',m,m,n);
        end
        
        %Find the solution with m clusters.
        [Cluster_elem,Clustering_error]=Weighted_Kernel_K_Means(Cluster_elem,K,Dataset_Weights,m,Display);
        
        if strcmp(Display,'details') || strcmp(Display,'nutshell')
            fprintf('\nFinal Clustering error=%f\n',Clustering_error);
        end
       
        %Keep the best solution with m clusters i.e. the one with lowest clustering error.
        if Best_error(m)>Clustering_error
            Best_error(m)=Clustering_error;
            Best_clusters(:,m)=Cluster_elem;
        end
    end
   
    if size(unique(Best_clusters(:,m)),1)<m
        error('Not able to find more than %d clusters\n',m-1);
    end
end

%Keep as final solution of Global Kernel K-Means the one with the lowest clustering error (usually its the one with Total_clusters).
[Clustering_error,Clusters]=min(Best_error);
Cluster_elem=Best_clusters(:,Clusters);

if strcmp(Display,'details') || strcmp(Display,'nutshell')
    fprintf('++++++++++++++++++++++++++++++++++++++++++++++++++++\n');
    fprintf('Best fit:%d clusters with Clustering Error=%f\n',Clusters,Clustering_error);
    fprintf('++++++++++++++++++++++++++++++++++++++++++++++++++++\n');
end
