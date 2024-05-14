

WITH
LPM_less12M_Accounts AS (
SELECT DISTINCT [AccountNumber]
     , max(mob) as max_mob               --,SnapshotDate
    FROM [Reporting-db].[nystart].[LoanPortfolioMonthly]
    WHERE  mob <= 12 
    GROUP by AccountNumber   ),


thirtyAtThree AS (
SELECT DISTINCT [AccountNumber]
     ,Ever30
    FROM [Reporting-db].[nystart].[LoanPortfolioMonthly]
    WHERE  mob = 3 
      ),

sixtyAtSix AS (
SELECT DISTINCT [AccountNumber]
     ,Ever60
    FROM [Reporting-db].[nystart].[LoanPortfolioMonthly]
    WHERE  mob = 6
      ),

DEL90 AS (
    SELECT DISTINCT L.AccountNumber ,T.Ever30 ,S.Ever60 ,L.Ever90 ,L.DisbursedDate ,L.MOB      --,SnapshotDate  
    FROM [Reporting-db].[nystart].[LoanPortfolioMonthly] as L
    inner join LPM_less12M_Accounts as A on L.AccountNumber =A.AccountNumber and L.MOB = A.max_mob  
    inner join thirtyAtThree as T on L.AccountNumber =T.AccountNumber 
    inner join sixtyAtSix as S on L.AccountNumber =S.AccountNumber 
    
     ) ,


apli as (

SELECT distinct
    
      a.[ReceivedDate]
      ,a.ApplicationID
      ,a.[AccountNumber]
        ,s.SSN
      ,a.[DisbursedDate]
      ,a.[Amount]
       ,a.[IsMainApplicant]
      ,a.[ApplicantNo]
      ,a.[HasCoapp]
      --,a.SSN


  FROM [Reporting-db].[nystart].[Applications] a


   left join [Reporting-db].[nystartSecure].[SsnMap] s  on a.SSN = s.ssn_hash and a.ApplicationID = s.ApplicationID

   where a.[Status] = 'DISBURSED'

   and s.SSN is not null

   and  a.ReceivedDate > '2022-02-01' and a.ReceivedDate < '2024-01-01'


) 

SELECT a.*     ,d.Ever90

from apli as a 

INNER JOIN DEL90 as d on a.AccountNumber = d.AccountNumber

--where d.AccountNumber = 7983117

order by AccountNumber








-- SELECT distinct
    
--       a.[ReceivedDate]
--       ,a.ApplicationID
--       ,a.[AccountNumber]
--         ,s.SSN
--         ,a.[Status]
--       ,a.[DisbursedDate]
--       ,a.[Amount]
--        ,a.[IsMainApplicant]
--       ,a.[ApplicantNo]
--       ,a.[HasCoapp]
--       --,a.SSN


--   FROM [Reporting-db].[nystart].[Applications] a


--    left join [Reporting-db].[nystartSecure].[SsnMap] s  on a.SSN = s.ssn_hash and a.ApplicationID = s.ApplicationID

--    where 
   
--    a.[Status] = 'DISBURSED'

--    and  s.SSN is not null and s.SSN = '9308240192'


  
-- 0003265501 0003165354 9204213467






-- WITH
-- LPM_less12M_Accounts AS (
-- SELECT DISTINCT [AccountNumber]
--      , max(mob) as max_mob               --,SnapshotDate
--     FROM [Reporting-db].[nystart].[LoanPortfolioMonthly]
--     WHERE  mob <= 12 
--     GROUP by AccountNumber   ),


-- thirtyAtThree AS (
-- SELECT DISTINCT [AccountNumber]
--      ,Ever30
--     FROM [Reporting-db].[nystart].[LoanPortfolioMonthly]
--     WHERE  mob = 3 
--       ),

-- sixtyAtSix AS (
-- SELECT DISTINCT [AccountNumber]
--      ,Ever60
--     FROM [Reporting-db].[nystart].[LoanPortfolioMonthly]
--     WHERE  mob = 6
--       ),

-- DEL90 AS (
--     SELECT DISTINCT L.AccountNumber ,T.Ever30 ,S.Ever60 ,L.Ever90 ,L.DisbursedDate ,L.MOB      --,SnapshotDate  
--     FROM [Reporting-db].[nystart].[LoanPortfolioMonthly] as L
--     left join LPM_less12M_Accounts as A on L.AccountNumber =A.AccountNumber and L.MOB = A.max_mob  
--     left join thirtyAtThree as T on L.AccountNumber =T.AccountNumber 
--     left join sixtyAtSix as S on L.AccountNumber =S.AccountNumber 
    
--      ) 


--      select * from DEL90 where AccountNumber = 7982291




