/* This determinant use Doolittle LU decomposition to calculate determinant 
   det(mat) = det(L)det(U) = det(U) = d1*d2*....*dk, where dk are the diagonal entries. */

double determinant(int dim, double** mat){
    // Gaussian elimination
    double ratio = 0;
    for(int i=0;i<dim;i++){
		if(mat[i][i] == 0.0){
		    printf("Mathematical Error!");
		    exit(0);
	    }
		for(int j=i+1;j<dim;j++){
			ratio = mat[j][i]/mat[i][i];
			for(int k=i; k<dim; k++){
			  	mat[j][k] = mat[j][k] - ratio*mat[i][k];
			}
		}
	}

    double det = 1;
    for(int i=0; i<dim; i++){
        det = det* mat[i][i];
    }
    return det;
}


// Default requirement: mat is squre.
// 1. calculate cofactor(mat)
// 2. adj(mat) = C^T
// 3. inv(mat = adj(mat)/det(mat)
double** inverse(int dim, double** mat)
{
    double det = determinant(dim, mat);
    if(det == 0.0){
        printf("Inverse of this matrix is impossible!\n");
        return NULL;
    }
    else{
        // 1. calculate cofactor(mat)
        double** cofactor = mat_zeros(dim, dim);
        // cofactor_ij = (-1)^(i+j) * Minor_ij
        for(int i=0; i<dim; i++){
            for(int j=0; j<dim; j++){
                // construct minor matrix of element A_ij, 
                // for purpose of calculating determinant
                double** minor = mat_zeros(dim-1, dim-1);
                for(int p=0; p<dim-1; p++){
                    for(int q=0; q<dim-1; q++){
                        if(p!=i && q==j){
                            minor[p][q] = mat[i][j+1];
                        }
                        if(p==i && q!=j){
                            minor[p][q] = mat[i+1][j];
                        }
                        if(p==i && q==j){
                            minor[p][q] = mat[i+1][j+1];
                        }
                        if(p!=i && q!=j){
                            minor[p][q] = mat[i][j];
                        }
                    }
                }
                cofactor[i][j] = (double)pow(-1, i+j)  * determinant(dim-1, minor);
                free_ptr(dim-1, minor);
            }
        }

        // 2. adj(mat) = C^T
        double** adjoint = mat_trans(dim, dim, cofactor);
        free_ptr(dim, cofactor);

        // 3. inv(mat = adj(mat)/det(mat)
        double** inv = mat_zeros(dim,dim);
        for(int i=0; i<dim; i++){
            for(int j=0; j<dim; j++){
                inv[i][j] = adjoint[i][j] / det;
            }
        }
        free_ptr(dim, adjoint);
        return inv;
    }
}
