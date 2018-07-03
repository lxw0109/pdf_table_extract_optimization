1. process_vd_hd()
2. pdftoppm -y 2400
3. maxdiv=10 -> maxdiv=25
4. splitTables: add splitTables(): split multiple tables in one page.
5. add parameter "offset" in splitTables(), because lthresh is too small, some Chinese word like "四" is considered to be a divider.

