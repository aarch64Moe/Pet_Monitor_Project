import 'dart:async';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pet Monitoring System',
      theme: ThemeData(
        fontFamily: 'Poppins',
        primaryColor: const Color(0xFF171F24),
        scaffoldBackgroundColor: const Color(0xFF121212),
        cardColor: const Color(0xFF1E282C),
        colorScheme: ColorScheme.dark().copyWith(
          primary: const Color(0xFF171F24),
          secondary: const Color(0xFF4BB2F9),
        ),
        textTheme: const TextTheme(
          bodyLarge: TextStyle(color: Colors.white),
          bodyMedium: TextStyle(color: Colors.white70),
        ),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> with SingleTickerProviderStateMixin {
  double temperature = 0.0;
  double humidity = 0.0;
  int heartRate = 0;
  String movement = 'Unknown';
  String classification = 'None';
  bool isRunning = false;
  bool isConnected = true;

  Timer? _timer;

  final List<Map<String, dynamic>> dataQueue = [];

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(const Duration(seconds: 5), (timer) => fetchData());
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    FocusScope.of(context).unfocus();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> fetchData() async {
    try {
      final response = await http.get(Uri.parse('http://192.168.1.209:5000/status'));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        if (mounted) {
          setState(() {
            temperature = data['temperature'] ?? 0.0;
            humidity = data['humidity'] ?? 0.0;
            heartRate = data['heart_rate'] ?? 0;
            movement = data['movement'] ?? 'Unknown';
            classification = data['classification'] ?? 'None';
            isRunning = data['is_running'] ?? false;
            isConnected = true;

            dataQueue.add({
              'movement': movement,
              'classification': classification,
              'timestamp': DateTime.now(),
            });

            if (dataQueue.length > 12) {
              dataQueue.removeAt(0);
            }
          });
        }
      } else {
        setDisconnected();
      }
    } catch (e) {
      setDisconnected();
    }
  }

  void setDisconnected() {
    if (mounted) {
      setState(() {
        isConnected = false;
        temperature = 0.0;
        humidity = 0.0;
        heartRate = 0;
        movement = 'Disconnected';
        classification = 'N/A';
        isRunning = false;
      });
    }
  }

  Map<String, dynamic> calculateReport() {
    Map<String, int> movementCount = {};
    Map<String, int> classificationCount = {};

    for (var entry in dataQueue) {
      movementCount[entry['movement']] = (movementCount[entry['movement']] ?? 0) + 1;
      classificationCount[entry['classification']] = (classificationCount[entry['classification']] ?? 0) + 1;
    }

    String mostCommonMovement = movementCount.entries.reduce((a, b) => a.value > b.value ? a : b).key;
    String mostCommonClassification = classificationCount.entries.reduce((a, b) => a.value > b.value ? a : b).key;

    return {
      'mostCommonMovement': mostCommonMovement,
      'mostCommonClassification': mostCommonClassification,
      'movementCount': movementCount[mostCommonMovement],
      'classificationCount': classificationCount[mostCommonClassification],
      'dataPoints': dataQueue.length,
    };
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pet Monitoring System', style: TextStyle(color: Colors.white)),
        backgroundColor: Theme.of(context).primaryColor,
        actions: [
          PopupMenuButton<String>(
            onSelected: (value) {
              if (value == 'Reboot') {
                rebootDevice();
              }
            },
            icon: const Icon(Icons.more_vert, color: Colors.white),
            itemBuilder: (BuildContext context) {
              return [
                PopupMenuItem<String>(
                  value: 'Reboot',
                  child: Row(
                    children: const [
                      Icon(Icons.restart_alt, color: Colors.redAccent),
                      SizedBox(width: 8),
                      Text('Reboot Device', style: TextStyle(color: Colors.redAccent)),
                    ],
                  ),
                ),
              ];
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              isConnected ? 'Device Connected' : 'Device Disconnected',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: isConnected ? Colors.green : Colors.red,
              ),
            ),

            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildInfoCard('Temperature', '${temperature.toStringAsFixed(1)}°C', Icons.thermostat, Colors.orange),
                _buildInfoCard('Humidity', '${humidity.toStringAsFixed(1)}%', Icons.water_drop, Colors.blue),
                _buildInfoCard('Heart Rate', '$heartRate BPM', Icons.favorite, Colors.red),
              ],
            ),

            Column(
              children: [
                RotatingIcon(isRunning: isRunning),
                const SizedBox(height: 10),
                Text(
                  isRunning ? 'AI Live Status: Running' : 'AI Live Status: Not Running',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: isRunning ? Colors.green : Colors.red,
                  ),
                ),
              ],
            ),

            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildInfoCard('Movement', movement, Icons.directions_walk, Colors.indigo),
                _buildInfoCard('Classification', classification, Icons.category, Theme.of(context).colorScheme.secondary),
              ],
            ),

            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildActionButton('Camera', Icons.camera_alt),
                _buildActionButton('Track GPS', Icons.gps_fixed),
                ElevatedButton.icon(
                  onPressed: () {
                    final report = calculateReport();
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => MonitorReportScreen(report: report),
                      ),
                    );
                  },
                  icon: const Icon(Icons.analytics),
                  label: const Text('Monitor'),
                  style: ElevatedButton.styleFrom(
                    textStyle: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.bold,
                      inherit: true,
                    ),
                    backgroundColor: Theme.of(context).colorScheme.secondary.withOpacity(0.9),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoCard(String title, String data, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(8),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        children: [
          Icon(icon, size: 30, color: color),
          const SizedBox(height: 8),
          Text(
            title,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.white),
          ),
          const SizedBox(height: 4),
          Text(
            data,
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: color),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton(String label, IconData icon) {
    return ElevatedButton.icon(
      onPressed: () {
        FocusScope.of(context).unfocus();
      },
      icon: Icon(icon, size: 18),
      label: Text(label),
      style: ElevatedButton.styleFrom(
        textStyle: const TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.bold,
          inherit: true,
        ),
        backgroundColor: Theme.of(context).colorScheme.secondary.withOpacity(0.9),
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    );
  }

  Future<void> rebootDevice() async {
    // Reboot logic
  }
}

class RotatingIcon extends StatefulWidget {
  final bool isRunning;

  const RotatingIcon({Key? key, required this.isRunning}) : super(key: key);

  @override
  _RotatingIconState createState() => _RotatingIconState();
}

class _RotatingIconState extends State<RotatingIcon> with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(seconds: 2));
    if (widget.isRunning) {
      _controller.repeat();
    }
  }

  @override
  void didUpdateWidget(RotatingIcon oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isRunning) {
      _controller.repeat();
    } else {
      _controller.stop();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return RotationTransition(
      turns: _controller,
      child: Icon(Icons.refresh, size: 40, color: widget.isRunning ? Colors.green : Colors.red),
    );
  }
}

class MonitorReportScreen extends StatelessWidget {
  final Map<String, dynamic> report;

  const MonitorReportScreen({Key? key, required this.report}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Monitor Report'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Most Common Movement: ${report['mostCommonMovement']}"),
            Text("Most Common Classification: ${report['mostCommonClassification']}"),
            Text("Movement Count: ${report['movementCount']}"),
            Text("Classification Count: ${report['classificationCount']}"),
            Text("Data Points: ${report['dataPoints']}"),
          ],
        ),
      ),
    );
  }
}