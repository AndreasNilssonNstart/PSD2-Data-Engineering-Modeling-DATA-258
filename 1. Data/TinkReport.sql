SELECT TOP (1000) [AccountJsonID]
      ,[TransactionJsonID]
      ,[ReportDate]
      ,[SSN]
      ,[AccountJSON]
      ,[TransactionJSON]
      ,[CheckID]
      ,[QueryString]
  FROM [nim-prod].[tink].[TinkReports]

  where SSN = '199202223989'