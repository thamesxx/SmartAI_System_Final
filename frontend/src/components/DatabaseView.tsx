import { useState, useEffect } from 'react';
import { getDatabaseRecords, DataRecord } from '../services/api';
import { toast } from 'sonner@2.0.3';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from './ui/table';
import { Search, Database, Download, Printer } from 'lucide-react';

const STATUS_BADGE: Record<string, string> = {
  running:     'text-emerald-700 bg-emerald-50 border border-emerald-200',
  idle:        'text-amber-700 bg-amber-50 border border-amber-200',
  maintenance: 'text-orange-700 bg-orange-50 border border-orange-200',
};

export function DatabaseView() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [records, setRecords] = useState<DataRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchRecords = async () => {
    try {
      const data = await getDatabaseRecords({ search: searchTerm, status: filterStatus });
      setRecords(data);
      setIsLoading(false);
    } catch {
      toast.error('Failed to fetch database records.');
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, [searchTerm, filterStatus]);

  const handleExportCSV = () => {
    const headers = [
      'ID', 'Timestamp', 'Machine', 'Lot 1', 'Lot 2', 'Article',
      'Length (m)', 'Speed (m/min)', 'Lot Time (min)', 'Run Time (min)',
      'SF (m³)', 'Water (L)', 'Air (Nm³)', 'Gas (m³)', 'Power (kWh)', 'Status',
    ];
    const csvContent = [
      headers.join(','),
      ...records.map((r) =>
        [
          r.id, r.timestamp, r.machineId, r.lot1, r.lot2, r.articleNumber,
          r.totalLength, r.speed, r.lotTime, r.machineRunningTime,
          r.sfConsumption, r.waterConsumption, r.airConsumption,
          r.gasConsumption, r.powerConsumption, r.status,
        ].join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `jeans-production-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('CSV exported successfully');
  };

  return (
    <div className="space-y-5">
      {/* Page title */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-gray-900">Production Database</h2>
          <p className="text-xs text-gray-500 mt-0.5">{records.length} records available</p>
        </div>
        <div className="flex items-center gap-2 print:hidden">
          <Button
            variant="outline"
            size="sm"
            onClick={handleExportCSV}
            className="gap-1.5 border-gray-200 text-gray-600 hover:bg-gray-50 text-xs"
          >
            <Download className="w-3.5 h-3.5" />
            Export CSV
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => window.print()}
            className="gap-1.5 border-gray-200 text-gray-600 hover:bg-gray-50 text-xs"
          >
            <Printer className="w-3.5 h-3.5" />
            Print
          </Button>
        </div>
      </div>

      {/* Main card */}
      <div className="rounded-xl border border-gray-200 bg-white overflow-hidden shadow-sm">
        {/* Filters */}
        <div className="flex flex-col md:flex-row gap-3 p-4 border-b border-gray-100 print:hidden">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400" />
            <Input
              placeholder="Search by lot, machine, or article…"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9 h-8 bg-gray-50 border-gray-200 text-gray-700 placeholder:text-gray-400 text-xs"
            />
          </div>
          <Select value={filterStatus} onValueChange={setFilterStatus}>
            <SelectTrigger className="w-36 h-8 bg-gray-50 border-gray-200 text-gray-600 text-xs">
              <SelectValue placeholder="All Status" />
            </SelectTrigger>
            <SelectContent className="bg-white border-gray-200">
              <SelectItem value="all" className="text-gray-700 text-xs">All Status</SelectItem>
              <SelectItem value="running" className="text-gray-700 text-xs">Running</SelectItem>
              <SelectItem value="idle" className="text-gray-700 text-xs">Idle</SelectItem>
              <SelectItem value="maintenance" className="text-gray-700 text-xs">Maintenance</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Table */}
        {isLoading ? (
          <div className="flex items-center justify-center py-16">
            <div className="w-6 h-6 border-2 border-gray-200 border-t-gray-600 rounded-full animate-spin" />
          </div>
        ) : (
          <div className="overflow-x-auto max-h-[calc(100vh-300px)] overflow-y-auto">
            <Table>
              <TableHeader className="sticky top-0 z-10 bg-gray-50">
                <TableRow className="border-gray-100 hover:bg-transparent">
                  {[
                    'ID', 'Timestamp', 'Machine', 'Lot 1', 'Lot 2', 'Article',
                    'Length', 'Speed', 'Lot Time', 'Run Time',
                    'SF (m³)', 'Water (L)', 'Air (Nm³)', 'Gas (m³)', 'Power (kWh)', 'Status',
                  ].map((h) => (
                    <TableHead key={h} className="text-gray-500 text-xs py-2 px-3 whitespace-nowrap">
                      {h}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {records.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={16} className="text-center text-gray-400 py-12">
                      No records found
                    </TableCell>
                  </TableRow>
                ) : (
                  records.map((r) => (
                    <TableRow
                      key={r.id}
                      className="border-gray-50 hover:bg-gray-50 transition-colors"
                    >
                      <TableCell className="text-xs font-mono text-gray-400 py-2 px-3">{r.id}</TableCell>
                      <TableCell className="text-xs text-gray-400 py-2 px-3 whitespace-nowrap">
                        {new Date(r.timestamp).toLocaleString()}
                      </TableCell>
                      <TableCell className="text-xs text-gray-700 py-2 px-3 whitespace-nowrap">{r.machineId}</TableCell>
                      <TableCell className="text-xs font-mono text-gray-500 py-2 px-3">{r.lot1}</TableCell>
                      <TableCell className="text-xs font-mono text-gray-500 py-2 px-3">{r.lot2}</TableCell>
                      <TableCell className="text-xs text-gray-700 py-2 px-3 whitespace-nowrap">{r.articleNumber}</TableCell>
                      <TableCell className="text-xs font-mono text-gray-700 py-2 px-3">{r.totalLength}m</TableCell>
                      <TableCell className="text-xs font-mono text-gray-700 py-2 px-3">{r.speed}</TableCell>
                      <TableCell className="text-xs font-mono text-gray-500 py-2 px-3">{r.lotTime}m</TableCell>
                      <TableCell className="text-xs font-mono text-gray-500 py-2 px-3">{r.machineRunningTime}m</TableCell>
                      <TableCell className="text-xs font-mono text-gray-500 py-2 px-3">{r.sfConsumption}</TableCell>
                      <TableCell className="text-xs font-mono text-gray-500 py-2 px-3">{r.waterConsumption}</TableCell>
                      <TableCell className="text-xs font-mono text-gray-500 py-2 px-3">{r.airConsumption}</TableCell>
                      <TableCell className="text-xs font-mono text-gray-500 py-2 px-3">{r.gasConsumption}</TableCell>
                      <TableCell className="text-xs font-mono text-gray-500 py-2 px-3">{r.powerConsumption}</TableCell>
                      <TableCell className="py-2 px-3">
                        <span className={`text-xs px-1.5 py-0.5 rounded ${STATUS_BADGE[r.status] ?? 'text-gray-500 bg-gray-50 border border-gray-200'}`}>
                          {r.status}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        )}

        {/* Footer */}
        <div className="px-4 py-2.5 border-t border-gray-100 flex items-center justify-between bg-gray-50">
          <span className="text-xs text-gray-400">
            Showing {records.length} records
          </span>
          <div className="flex items-center gap-1.5 text-xs text-gray-400">
            <Database className="w-3 h-3" />
            <span>Jeans Production DB</span>
          </div>
        </div>
      </div>
    </div>
  );
}
