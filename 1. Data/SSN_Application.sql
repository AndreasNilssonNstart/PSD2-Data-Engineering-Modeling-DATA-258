
with apli as (

SELECT distinct
    
      a.[IsMainApplicant]
      ,a.[ApplicantNo]
      ,a.[HasCoapp]
      ,a.[ReceivedDate]
      ,a.ApplicationID
      ,a.[AccountNumber]
      ,a.[DisbursedDate]
      ,a.[Amount]
      --,a.SSN
      ,s.SSN

  FROM [Reporting-db].[nystart].[Applications] a


   left join [Reporting-db].[nystartSecure].[SsnMap] s  on a.SSN = s.ssn_hash and a.ApplicationID = s.ApplicationID

   where a.[Status] = 'DISBURSED'

   and s.SSN is not null

   and a.ReceivedDate > '2022-03-01'


) 

SELECT * from apli

order by SSN


