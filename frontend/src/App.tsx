import { useState } from 'react';
import { Upload, FileText, TrendingUp, TrendingDown, DollarSign, Calendar, Globe, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, Area, AreaChart } from 'recharts';
import { useDropzone } from 'react-dropzone';

// Sample data for demonstration
const sampleData = {
  accountInfo: {
    customerName: "John Smith",
    accountNumber: "ACC-2024-001",
    ibanNumber: "GB29 NWBK 6016 1331 9268 19",
    openingBalance: 15750.00,
    closingBalance: 18420.50,
    financialPeriod: "Jan 2024 - Dec 2024",
    pagesProcessed: 156,
    totalTransactions: 2847
  },
  monthlyData: [
    { month: 'Jan', openingBalance: 15750, closingBalance: 16200, inflows: 8500, outflows: 8050, netChange: 450, fluctuation: 1200, foreignTxns: 12, foreignAmount: 2400, minBalance: 14800, overdraftDays: 0 },
    { month: 'Feb', openingBalance: 16200, closingBalance: 15950, inflows: 7800, outflows: 8050, netChange: -250, fluctuation: 1100, foreignTxns: 8, foreignAmount: 1600, minBalance: 15200, overdraftDays: 0 },
    { month: 'Mar', openingBalance: 15950, closingBalance: 17100, inflows: 9200, outflows: 8050, netChange: 1150, fluctuation: 1400, foreignTxns: 15, foreignAmount: 3200, minBalance: 15100, overdraftDays: 0 },
    { month: 'Apr', openingBalance: 17100, closingBalance: 16800, inflows: 8100, outflows: 8400, netChange: -300, fluctuation: 900, foreignTxns: 6, foreignAmount: 980, minBalance: 16200, overdraftDays: 0 },
    { month: 'May', openingBalance: 16800, closingBalance: 18200, inflows: 9800, outflows: 8400, netChange: 1400, fluctuation: 1600, foreignTxns: 18, foreignAmount: 4100, minBalance: 16200, overdraftDays: 0 },
    { month: 'Jun', openingBalance: 18200, closingBalance: 17900, inflows: 8600, outflows: 8900, netChange: -300, fluctuation: 1300, foreignTxns: 11, foreignAmount: 2200, minBalance: 17200, overdraftDays: 0 }
  ],
  analytics: {
    averageFluctuation: 1250,
    netCashFlowStability: 85.2,
    totalForeignTransactions: 70,
    totalForeignAmount: 14480,
    overdraftFrequency: 2,
    overdraftTotalDays: 8
  }
};

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'];

function mapAccountInfo(accountInfo: any) {
  return {
    customerName: accountInfo.customer_name || "",
    accountNumber: accountInfo.account_number || "",
    ibanNumber: accountInfo.iban_number || "",
    openingBalance: accountInfo.opening_balance || 0,
    closingBalance: accountInfo.closing_balance || 0,
    financialPeriod: accountInfo.financial_period || "",
    pagesProcessed: accountInfo.pages_processed || 0,
    totalTransactions: accountInfo.total_transactions || 0,
  };
}

function mapMonthlyStats(month: string, stats: any) {
  return {
    month,
    openingBalance: stats.opening_balance ?? 0,
    closingBalance: stats.closing_balance ?? 0,
    inflows: stats.total_credit ?? 0,
    outflows: Math.abs(stats.total_debit ?? 0),
    netChange: stats.net_change ?? 0,
    fluctuation: stats.fluctuation ?? 0,
    foreignTxns: (stats.international_inward_count ?? 0) + (stats.international_outward_count ?? 0),
    foreignAmount: (stats.international_inward_total ?? 0) + (stats.international_outward_total ?? 0),
    minBalance: stats.minimum_balance ?? 0,
  };
}

function App() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [accountInfo, setAccountInfo] = useState<any | null>(null);
  const [monthlyData, setMonthlyData] = useState<any[]>([]);
  const [analytics, setAnalytics] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    maxSize: 2 * 1024 * 1024, // 2MB in bytes
    onDrop: async (acceptedFiles, fileRejections) => {
      // Handle rejected files (e.g., due to size or type)
      if (fileRejections.length > 0) {
        const rejectionErrors = fileRejections[0].errors;
        if (rejectionErrors.some((err) => err.code === 'file-too-large')) {
          setError('File is too large. Please upload a PDF file smaller than 2MB.');
        } else if (rejectionErrors.some((err) => err.code === 'file-invalid-type')) {
          setError('Invalid file type. Please upload a PDF file.');
        } else {
          setError('An error occurred while uploading the file.');
        }
        return;
      }

      if (acceptedFiles.length > 0) {
        setUploadedFile(acceptedFiles[0]);
        setIsAnalyzing(true);
        setError(null);
        setShowResults(false);
        const formData = new FormData();
        formData.append('file', acceptedFiles[0]);
        try {
          const uploadUrl = 'http://localhost:8000/api/pdf/upload/';
          console.log('Uploading to:', uploadUrl);

          const startTime = performance.now();

          const response = await fetch(uploadUrl, {
            method: 'POST',
            body: formData
          });
          const endTime = performance.now();
          console.log(`Upload + server processing took ${(endTime - startTime).toFixed(2)} ms`);
          if (!response.ok) {
            const err = await response.json();
            setError(err.error || 'Failed to process PDF');
            setIsAnalyzing(false);
            return;
          }
          const data = await response.json();
          console.log(data);
          setAccountInfo(mapAccountInfo(data.account_info || {}));
          setMonthlyData(data.monthly_analysis ? Object.entries(data.monthly_analysis).map(([month, stats]) => mapMonthlyStats(month, stats)) : []);
          setAnalytics({
            averageFluctuation: data.analytics?.average_fluctuation,
            netCashFlowStability: data.analytics?.net_cash_flow_stability,
            totalForeignTransactions: data.analytics?.total_foreign_transactions,
            totalForeignAmount: data.analytics?.total_foreign_amount,
            overdraftFrequency: data.analytics?.overdraft_frequency,
            overdraftTotalDays: data.analytics?.overdraft_total_days,
            sum_total_inflow: data.analytics?.sum_total_inflow,
            sum_total_outflow: data.analytics?.sum_total_outflow,
            avg_total_inflow: data.analytics?.avg_total_inflow,
            avg_total_outflow: data.analytics?.avg_total_outflow,
          });
          setIsAnalyzing(false);
          setShowResults(true);
        } catch (e) {
          setError('Network or server error');
          setIsAnalyzing(false);
        }
      }
    },
  });

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'SAR'
    }).format(amount);
  };

  const StatCard = ({ title, value, icon: Icon, trend, color = "text-blue-600" }: any) => (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
          {trend && (
            <div className={`flex items-center mt-2 ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
              {trend > 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
              <span className="text-sm ml-1">{Math.abs(trend)}%</span>
            </div>
          )}
        </div>
        <Icon className={`w-8 h-8 ${color}`} />
      </div>
    </div>
  );

  if (!showResults) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="container mx-auto px-4 py-12">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h1 className="text-4xl font-bold text-gray-900 mb-4">Financial PDF Analyzer</h1>
              <p className="text-lg text-gray-600">Upload your bank statement PDF for comprehensive financial analysis (Max 2MB)</p>
            </div>

            <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
              {error && (
                <div className="mb-4 p-4 bg-red-50 text-red-700 rounded-lg flex items-center">
                  <AlertTriangle className="w-5 h-5 mr-2" />
                  {error}
                </div>
              )}
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200 ${
                  isDragActive
                    ? 'border-blue-500 bg-blue-50'
                    : uploadedFile
                    ? 'border-green-500 bg-green-50'
                    : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
                }`}
              >
                <input {...getInputProps()} />
                {isAnalyzing ? (
                  <div className="flex flex-col items-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                    <p className="text-lg font-semibold text-gray-700">Analyzing PDF...</p>
                    <p className="text-sm text-gray-500">Processing your financial data</p>
                  </div>
                ) : uploadedFile ? (
                  <div className="flex flex-col items-center">
                    <FileText className="w-12 h-12 text-green-600 mb-4" />
                    <p className="text-lg font-semibold text-gray-700">{uploadedFile.name}</p>
                    <p className="text-sm text-gray-500">File uploaded successfully</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <Upload className="w-12 h-12 text-gray-400 mb-4" />
                    <p className="text-lg font-semibold text-gray-700">
                      {isDragActive ? 'Drop your PDF here' : 'Drag & drop your PDF here'}
                    </p>
                    <p className="text-sm text-gray-500">or click to select a file (Max 2MB)</p>
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div className="flex items-center mb-4">
                  <FileText className="w-6 h-6 text-blue-600 mr-2" />
                  <h3 className="font-semibold text-gray-900">Account Overview</h3>
                </div>
                <p className="text-sm text-gray-600">Customer details, account numbers, and basic information</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div className="flex items-center mb-4">
                  <TrendingUp className="w-6 h-6 text-green-600 mr-2" />
                  <h3 className="font-semibold text-gray-900">Monthly Analysis</h3>
                </div>
                <p className="text-sm text-gray-600">Detailed breakdown of monthly transactions and patterns</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div className="flex items-center mb-4">
                  <DollarSign className="w-6 h-6 text-purple-600 mr-2" />
                  <h3 className="font-semibold text-gray-900">Analytics</h3>
                </div>
                <p className="text-sm text-gray-600">Financial stability metrics and insights</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Financial Analysis Report</h1>
              <p className="text-sm text-gray-600 mt-1">Generated from {uploadedFile?.name}</p>
            </div>
            <button
              onClick={() => setShowResults(false)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Upload New PDF
            </button>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {/* Account Overview */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
            <FileText className="w-6 h-6 text-blue-600 mr-2" />
            Account Overview
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div>
              <p className="text-sm font-medium text-gray-600">Customer Name</p>
              <p className="text-lg font-semibold text-gray-900">{accountInfo?.customerName}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">Account Number</p>
              <p className="text-lg font-semibold text-gray-900">{accountInfo?.accountNumber}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">IBAN Number</p>
              <p className="text-lg font-semibold text-gray-900">{accountInfo?.ibanNumber}</p>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">Financial Period</p>
              <p className="text-lg font-semibold text-gray-900">{accountInfo?.financialPeriod}</p>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
            <StatCard
              title="Opening Balance"
              value={formatCurrency(accountInfo?.openingBalance || 0)}
              icon={TrendingUp}
              color="text-green-600"
            />
            <StatCard
              title="Closing Balance"
              value={formatCurrency(accountInfo?.closingBalance || 0)}
              icon={TrendingUp}
              color="text-blue-600"
            />
            <StatCard
              title="Pages Processed"
              value={accountInfo?.pagesProcessed || 0}
              icon={FileText}
              color="text-purple-600"
            />
            <StatCard
              title="Total Transactions"
              value={accountInfo?.totalTransactions?.toLocaleString() || 0}
              icon={DollarSign}
              color="text-orange-600"
            />
          </div>
        </div>

        {/* Monthly Analysis */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
            <Calendar className="w-6 h-6 text-blue-600 mr-2" />
            Monthly Analysis
          </h2>

          {/* 1. Overall Numbers Subsection */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Overall Numbers</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-4">
              <StatCard
                title="Sum of Total Inflow"
                value={formatCurrency(analytics?.sum_total_inflow || 0)}
                icon={TrendingUp}
                color="text-green-600"
              />
              <StatCard
                title="Sum of Total Outflow"
                value={formatCurrency(analytics?.sum_total_outflow || 0)}
                icon={TrendingDown}
                color="text-red-600"
              />
              <StatCard
                title="Average Total Inflow"
                value={formatCurrency(analytics?.avg_total_inflow || 0)}
                icon={TrendingUp}
                color="text-green-600"
              />
              <StatCard
                title="Average Total Outflow"
                value={formatCurrency(analytics?.avg_total_outflow || 0)}
                icon={TrendingDown}
                color="text-red-600"
              />
              <StatCard
                title="Average Fluctuation"
                value={analytics?.averageFluctuation !== undefined ? analytics.averageFluctuation.toFixed(2) + '%' : '0.00%'}
                icon={TrendingUp}
                color="text-blue-600"
              />
              <StatCard
                title="Cash Flow Stability"
                value={analytics?.netCashFlowStability !== undefined ? analytics.netCashFlowStability.toFixed(4) : '0.0000'}
                icon={TrendingUp}
                color="text-green-600"
              />
              <StatCard
                title="Foreign Transactions"
                value={`${analytics?.totalForeignTransactions || 0} (${formatCurrency(analytics?.totalForeignAmount || 0)})`}
                icon={Globe}
                color="text-purple-600"
              />
              <StatCard
                title="Overdraft Events"
                value={`${analytics?.overdraftFrequency || 0} times (${analytics?.overdraftTotalDays || 0} days)`}
                icon={AlertTriangle}
                color="text-red-600"
              />
            </div>
          </div>

          {/* 2. Charts Subsection */}
          <div className="mb-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Balance Trend</h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={monthlyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip formatter={(value) => [formatCurrency(value as number), '']} />
                  <Area type="monotone" dataKey="openingBalance" stackId="1" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="closingBalance" stackId="2" stroke="#10B981" fill="#10B981" fillOpacity={0.6} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Monthly Cash Flow</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={monthlyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip formatter={(value) => [formatCurrency(value as number), '']} />
                  <Bar dataKey="inflows" fill="#10B981" />
                  <Bar dataKey="outflows" fill="#EF4444" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Monthly Fluctuation</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={monthlyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip formatter={(value) => [typeof value === 'number' ? value.toFixed(2) + '%' : Number(value).toFixed(2) + '%', 'Fluctuation']} />
                  <Line type="monotone" dataKey="fluctuation" stroke="#8B5CF6" strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Foreign Transaction Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={monthlyData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ month, foreignTxns }) => `${month}: ${foreignTxns}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="foreignTxns"
                  >
                    {monthlyData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 3. Monthly Details Table Subsection */}
          <div className="overflow-x-auto">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Monthly Details</h3>
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Month</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Opening Balance</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Closing Balance</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Inflows</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Outflows</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Net Change</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Fluctuation</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Foreign Txns</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700">Min Balance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {monthlyData.map((month, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-medium text-gray-900">{month.month}</td>
                    <td className="px-4 py-3 text-gray-700">{formatCurrency(month.openingBalance)}</td>
                    <td className="px-4 py-3 text-gray-700">{formatCurrency(month.closingBalance)}</td>
                    <td className="px-4 py-3 text-green-600">{formatCurrency(month.inflows)}</td>
                    <td className="px-4 py-3 text-red-600">{formatCurrency(month.outflows)}</td>
                    <td className={`px-4 py-3 ${month.netChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>{formatCurrency(month.netChange)}</td>
                    <td className="px-4 py-3 text-gray-700">{month.fluctuation.toFixed(2)}%</td>
                    <td className="px-4 py-3 text-gray-700">{month.foreignTxns} ({formatCurrency(month.foreignAmount)})</td>
                    <td className="px-4 py-3 text-gray-700">{formatCurrency(month.minBalance)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Overall Analytics */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
            <TrendingUp className="w-6 h-6 text-blue-600 mr-2" />
            Financial Analytics
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <StatCard
              title="Average Fluctuation"
              value={analytics?.averageFluctuation !== undefined ? analytics.averageFluctuation.toFixed(2) + '%' : '0.00%'}
              icon={TrendingUp}
              color="text-blue-600"
            />
            <StatCard
              title="Cash Flow Stability"
              value={`${analytics?.netCashFlowStability || 0}%`}
              icon={TrendingUp}
              color="text-green-600"
            />
            <StatCard
              title="Foreign Transactions"
              value={`${analytics?.totalForeignTransactions || 0} (${formatCurrency(analytics?.totalForeignAmount || 0)})`}
              icon={Globe}
              color="text-purple-600"
            />
            <StatCard
              title="Overdraft Events"
              value={`${analytics?.overdraftFrequency || 0} times (${analytics?.overdraftTotalDays || 0} days)`}
              icon={AlertTriangle}
              color="text-red-600"
            />
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Monthly Fluctuation</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={monthlyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip formatter={(value) => [Number(value).toFixed(2) + '%', 'Fluctuation']} />
                  <Line type="monotone" dataKey="fluctuation" stroke="#8B5CF6" strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Foreign Transaction Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={monthlyData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ month, foreignTxns }) => `${month}: ${foreignTxns}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="foreignTxns"
                  >
                    {monthlyData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;