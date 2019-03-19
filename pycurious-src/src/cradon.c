
#include "cradon.h"

int radon2d(const double* Tx,
             const double* Rx,
             const size_t nTx,
             const size_t nx,
             const size_t ny,
             PyObject* L) {

        const double  small=1.e-10;

        size_t nCells = nx * ny;
        size_t nLmax = nTx * nx * ny/2;
        double percent_sp = (nLmax*1.0)/(nTx*nCells*1.0);

        double* data_p = (double*)malloc( nLmax*sizeof(double) );
        int64_t* indices_p = (int64_t*)malloc( nLmax*sizeof(int64_t) );
        int64_t* indptr_p = (int64_t*)malloc( (nTx+1)*sizeof(int64_t) );

        double* grx = (double*)malloc( (nx+1)*sizeof(double) );
        for ( size_t n=0; n<nx+1; ++n) {
            grx[n] = -0.5 + n;
            //printf("%f\n", grx[n]);
        }

        double* gry = (double*)malloc( (ny+1)*sizeof(double) );
        for ( size_t n=0; n<ny+1; ++n) {
            gry[n] = -0.5 + n;
            //printf("%f\n", gry[n]);
        }

        size_t k = 0;
        size_t ix, iy;
        for ( size_t n=0; n<nTx; ++n ) {
            indptr_p[n] = k;

            double xs = Tx[2*n];
            double ys = Tx[2*n+1];
            double xr = Rx[2*n];
            double yr = Rx[2*n+1];

            //printf("xs, ys, xr, yr = %f %f %f %f\n", xs, ys, xr, yr);

            //double l = sqrt( (xr-xs)*(xr-xs) + (yr-ys)*(yr-ys) );  /* longeur du rai */

            if ( xs>xr ) {  /* on va de s à r, on veut x croissant */
                double dtmp = xs;
                xs = xr;
                xr = dtmp;
                dtmp = ys;
                ys = yr;
                yr = dtmp;
                //printf("    xs, ys, xr, yr = %f %f %f %f\n", xs, ys, xr, yr);
            }

            /* points de depart */
            double x = xs;
            double y = ys;

            if ( fabs(ys-yr)<small ) {  /* rai horizontal */

                for ( ix=0; ix<nx; ++ix ) if ( x < grx[ix+1] ) break;
                for ( iy=0; iy<ny; ++iy ) if ( y < gry[iy+1] ) break;

                while ( x < xr ) {
                    int64_t iCell = ix*ny + iy;

                    double dlx = ( grx[ix+1]<xr ? grx[ix+1] : xr ) - x;

                    indices_p[k] = iCell;
                    data_p[k] = dlx;
                    k++;

                    if (k>=nLmax){
                        size_t oldnymax = nLmax;
                        percent_sp += 0.1;
                        nLmax = (size_t)ceil((double)nTx*(double)nCells*percent_sp);

                        /* make sure nzmax increases at least by 1 */
                        if (oldnymax == nLmax) nLmax++;

                        data_p = (double*)realloc( data_p, nLmax*sizeof(double) );
                        indices_p = (int64_t*)realloc( indices_p, nLmax*sizeof(int64_t) );
                    }

                    ix++;
                    x = grx[ix];
                }
            }
            else if ( fabs(xs-xr)<small ) { /* rai vertical */
                if ( ys > yr ) {  /* on va de s à r, on veut y croissant */
                    double dtmp = ys;
                    ys = yr;
                    yr = dtmp;
                }
                y = ys;

                for ( ix=0; ix<nx; ++ix ) if ( x < grx[ix+1] ) break;
                for ( iy=0; iy<ny; ++iy ) if ( y < gry[iy+1] ) break;

                while ( y < yr ) {
                    int64_t iCell = ix*ny + iy;

                    double dly = ( gry[iy+1]<yr ? gry[iy+1] : yr ) - y;

                    indices_p[k] = iCell;
                    data_p[k] = dly;
                    k++;

                    if (k>=nLmax){
                        size_t oldnymax = nLmax;
                        percent_sp += 0.1;
                        nLmax = (size_t)ceil((double)nTx*(double)nCells*percent_sp);

                        /* make sure nymax increases at least by 1 */
                        if (oldnymax == nLmax) nLmax++;

                        data_p = (double*)realloc( data_p, nLmax*sizeof(double) );
                        indices_p = (int64_t*)realloc( indices_p, nLmax*sizeof(int64_t) );
                    }

                    iy++;
                    y = gry[iy];
                }
            }
            else { /* rai oblique */
                /* pente du rai */
                double m = (yr-ys)/(xr-xs);
                double b = yr - m*xr;
                int up = m>0;

                for ( ix=0; ix<nx; ++ix ) if ( x < grx[ix+1] ) break;
                for ( iy=0; iy<ny; ++iy ) if ( y < gry[iy+1] ) break;

                while ( x < xr ) {

                    double yi = m*grx[ix+1] + b;

                    if ( up ) {
                        while ( y < yi && y < yr ) {
                            int64_t iCell = ix*ny + iy;

                            double ye = gry[iy+1]<yi ? gry[iy+1] : yi;
                            ye = ye<yr ? ye : yr;
                            double xe = (ye-b)/m;
                            double dlx = xe - x;
                            double dly = ye - y;
                            double dl = sqrt( dlx*dlx + dly*dly );


                            if (dl>1.5) {
                              printf("up - dl = %f, xe = %f, x = %f, ye = %f, y = %f\n", dl, xe, x, ye, y);
                              printf("     xs = %f, ys = %f, xr = %f, yr = %f\n", xs, ys, xr, yr);
                              printf("     m = %f, b = %f, yi = %f\n", m, b, yi);
                            }

                            indices_p[k] = iCell;
                            data_p[k] = dl;
                            k++;

                            if (k>=nLmax){
                                size_t oldnymax = nLmax;
                                percent_sp += 0.1;
                                nLmax = (size_t)ceil((double)nTx*(double)nCells*percent_sp);

                                /* make sure nymax increases at least by 1 */
                                if (oldnymax == nLmax) nLmax++;

                                data_p = (double*)realloc( data_p, nLmax*sizeof(double) );
                                indices_p = (int64_t*)realloc( indices_p, nLmax*sizeof(int64_t) );
                            }

                            x = xe;
                            y = ye;
                            if ( fabs(y-gry[iy+1])<small ) iy++;
                        }
                    } else {
                        while ( y > yi && y > yr ) {
                            int64_t iCell = ix*ny + iy;

                            double ye = gry[iy]>yi ? gry[iy] : yi;
                            ye = ye>yr ? ye : yr;
                            double xe = (ye-b)/m;
                            double dlx = xe - x;
                            double dly = ye - y;
                            double dl = sqrt( dlx*dlx + dly*dly );

                            if (dl>1.5) {
                              printf("down - dl = %f, %f %f %f %f\n", dl, xe, x, ye, y);
                              printf("     xs, ys, xr, yr = %f %f %f %f\n", xs, ys, xr, yr);
                              printf("     m = %f, b = %f, yi = %f\n", m, b, yi);
                            }
                            indices_p[k] = iCell;
                            data_p[k] = dl;
                            k++;

                            if (k>=nLmax){
                                size_t oldnymax = nLmax;
                                percent_sp += 0.1;
                                nLmax = (size_t)ceil((double)nTx*(double)nCells*percent_sp);

                                /* make sure nymax increases at least by 1 */
                                if (oldnymax == nLmax) nLmax++;

                                data_p = (double*)realloc( data_p, nLmax*sizeof(double) );
                                indices_p = (int64_t*)realloc( indices_p, nLmax*sizeof(int64_t) );
                            }

                            x = xe;
                            y = ye;
                            if ( fabs(y-gry[iy])<small ) iy--;
                        }
                    }

                    ix++;
                    x = grx[ix];
                }
            }
        }

        indptr_p[nTx] = k;
        size_t nnz = k;

        data_p = (double*)realloc( data_p, nnz*sizeof(double) );
        indices_p = (int64_t*)realloc( indices_p, nnz*sizeof(int64_t) );

        import_array();  // to use PyArray_SimpleNewFromData

        npy_intp dims[] = {(npy_intp)nnz};
        PyObject* data = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, data_p);
        PyArray_ENABLEFLAGS((PyArrayObject*)data, NPY_ARRAY_OWNDATA);
        PyObject* indices = PyArray_SimpleNewFromData(1, dims, NPY_INT64, indices_p);
        PyArray_ENABLEFLAGS((PyArrayObject*)indices, NPY_ARRAY_OWNDATA);
        dims[0] = nTx+1;
        PyObject* indptr = PyArray_SimpleNewFromData(1, dims, NPY_INT64, indptr_p);
        PyArray_ENABLEFLAGS((PyArrayObject*)indptr, NPY_ARRAY_OWNDATA);
        PyTuple_SetItem(L, 0, data);
        PyTuple_SetItem(L, 1, indices);
        PyTuple_SetItem(L, 2, indptr);

        free(grx);
        free(gry);

        return 0;
    }
