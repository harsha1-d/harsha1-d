package contest_2ima20.client.networking;

import contest_2ima20.core.problem.Problem;
import contest_2ima20.core.problem.Solution;
import contest_2ima20.core.util.Settings;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.security.cert.X509Certificate;
import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLHandshakeException;
import javax.net.ssl.SSLSession;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

public class Sender {

    private static boolean joined = false;

    public static String joinContest() {
        joined = false;
        String conn = Settings.getValue("connectionString", null);
        String team = Settings.getValue("teamName", null);
        String secret = Settings.getValue("sharedSecret", null);
        if (conn == null || conn.trim().isEmpty()) {
            return "Specify a host URL";
        }
        if (team == null || team.trim().isEmpty()) {
            return "Specify a team name";
        }
        if (secret == null || secret.trim().isEmpty()) {
            return "Specify a secret";
        }

        try {
            String responseString = postPlainText(conn, team + "\t" + secret);

            if (responseString != null && !responseString.isEmpty()) {
                System.err.println("Error: " + responseString);
                return responseString;
            }

            joined = true;
            return null;
        } catch (MalformedURLException ex) {
            return "Invalid contest URL. Include http:// or https://";
        } catch (UnknownHostException ex) {
            return "Could not find the contest server. Check the URL and your internet connection";
        } catch (SocketTimeoutException ex) {
            return "Timed out while contacting the contest server";
        } catch (ConnectException ex) {
            return "Could not connect to the contest server";
        } catch (SSLHandshakeException ex) {
            return "TLS certificate validation failed. The contest server certificate is not trusted by this Java runtime";
        } catch (ContestConnectionException ex) {
            return ex.getMessage();
        } catch (Exception ex) {
            ex.printStackTrace();
            return "Connection error occurred: " + ex.getMessage();
        }

    }

    public static void leaveContest() {
        joined = false;
    }

    public static void sendSolution(Problem p, Solution s) {
        if (joined) {
            String conn = Settings.getValue("connectionString", null);
            String team = Settings.getValue("teamName", null);
            String secret = Settings.getValue("sharedSecret", null);
            if (conn == null || conn.trim().isEmpty()) {
                return;
            }
            if (team == null || team.trim().isEmpty()) {
                return;
            }
            if (secret == null || secret.trim().isEmpty()) {
                return;
            }

            try {
                String responseString = postPlainText(conn,
                        team + "\t" + secret + "\t" + p.instanceName() + "\n" + s.write());

                System.out.println("" + responseString);

            } catch (ContestConnectionException ex) {
                System.err.println(ex.getMessage());
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static String getConnectionString() {
        return Settings.getValue("connectionString", "");
    }

    public static void setConnectionString(String connectionString) {
        Settings.setValue("connectionString", connectionString);
    }

    public static String getTeamName() {
        return Settings.getValue("teamName", "");
    }

    public static void setTeamName(String teamName) {
        Settings.setValue("teamName", teamName);
    }

    public static String getSharedSecret() {
        return Settings.getValue("sharedSecret", "");
    }

    public static void setSharedSecret(String sharedSecret) {
        if (sharedSecret != null) {
            sharedSecret = sharedSecret.trim();
        }
        Settings.setValue("sharedSecret", sharedSecret);
    }

    public static boolean isJoinedContest() {
        return joined;
    }

    private static String postPlainText(String connectionString, String body) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(connectionString.trim()).openConnection();
        if (connection instanceof HttpsURLConnection) {
            configureHttps((HttpsURLConnection) connection);
        }
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setConnectTimeout(5000);
        connection.setReadTimeout(5000);
        connection.setRequestProperty("Content-Type", "text/plain; charset=UTF-8");

        byte[] payload = body.getBytes(StandardCharsets.UTF_8);
        connection.setFixedLengthStreamingMode(payload.length);

        try (OutputStream output = connection.getOutputStream()) {
            output.write(payload);
        }

        int responseCode = connection.getResponseCode();
        InputStream stream = responseCode >= 400
                ? connection.getErrorStream()
                : connection.getInputStream();
        String responseBody = "";
        if (stream != null) {
            try (InputStream input = stream) {
                responseBody = new String(input.readAllBytes(), StandardCharsets.UTF_8);
            }
        }

        try {
            if (responseCode >= 400) {
                if (responseBody == null || responseBody.isBlank()) {
                    throw new ContestConnectionException("Contest server returned HTTP " + responseCode);
                }
                throw new ContestConnectionException(responseBody);
            }
            return responseBody;
        } finally {
            connection.disconnect();
        }
    }

    private static void configureHttps(HttpsURLConnection connection) throws Exception {
        if (!Settings.getBoolean("allowInsecureContestHttps", true)) {
            return;
        }

        TrustManager[] trustAllCerts = new TrustManager[]{
            new X509TrustManager() {
                @Override
                public void checkClientTrusted(X509Certificate[] chain, String authType) {
                }

                @Override
                public void checkServerTrusted(X509Certificate[] chain, String authType) {
                }

                @Override
                public X509Certificate[] getAcceptedIssuers() {
                    return new X509Certificate[0];
                }
            }
        };

        SSLContext sslContext = SSLContext.getInstance("TLS");
        sslContext.init(null, trustAllCerts, new SecureRandom());
        connection.setSSLSocketFactory(sslContext.getSocketFactory());
        HostnameVerifier trustAllHosts = new HostnameVerifier() {
            @Override
            public boolean verify(String hostname, SSLSession session) {
                return true;
            }
        };
        connection.setHostnameVerifier(trustAllHosts);
    }

    private static class ContestConnectionException extends Exception {

        ContestConnectionException(String message) {
            super(message);
        }
    }

}
