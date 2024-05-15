# PSD2-Data-Engineering-Modeling-DATA-258
PSD2 Data Engineering &amp; Modeling


## Environment management

### Create env

1. Navigate to project dir (e.g. C:\Users\Luis\Repositories\Secure-Key-Reporting--DB)

    ```cd C:\Users\Luis\Repositories\Secure-Key-Reporting--DB```

1. Create conda environment

    ```conda env create -f environment.yml```

### Update environment
1. Make sure environment is activated

    ```conda activate data258```

1. Install package(s):

    ```pip install numpy```

1. Export updated environment.yml:

    ```conda env export | findstr -v "prefix" > environment.yml```

