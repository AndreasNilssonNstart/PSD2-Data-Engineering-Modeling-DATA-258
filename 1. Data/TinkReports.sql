


with Account

as (

SELECT  [AccountJsonID]
      ,[TransactionJsonID]
      ,[ReportDate]
      ,[SSN]
      ,[AccountJSON]
      ,[TransactionJSON]
      ,[CheckID]
      ,[QueryString]
  FROM [nim-production-backup].[tink].[TinkReports]

 where SSN =  199912270692

)

,lastreport as (SELECT   max(ReportDate) as ReportDate from Account )


select * from Account as a inner join lastreport as l on a.ReportDate = l.ReportDate




